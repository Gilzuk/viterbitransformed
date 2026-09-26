from Code.channel.channel_estimation import estimate_channel
from Code.channel.modulator import BPSKModulator
from Code.channel.channel import ISIAWGNChannel
from Code.ecc.rs_main import encode
from Code.channel.data_cache import ChannelDataCache
from torch.utils.data import Dataset
from numpy.random import mtrand
from typing import Tuple, List
import numpy as np
import torch

import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(device)

# Global cache instance
_data_cache = ChannelDataCache()

# Only the first CACHE_MAX_REP repetitions of a point are persisted to disk.
# Each cached draw is ~0.24 MB, so a high-SNR point running 100k repetitions
# (needed to accumulate errors down at the error floor) would otherwise write
# ~23 GB of cache files and fill the disk. Caching the opening repetitions
# still gives the useful property -- different models evaluated at the same
# rep index see the same channel draw, so comparisons stay paired -- while
# the long tail is generated fresh and simply not persisted.
CACHE_MAX_REP = 200

# Fast fading (doppler > 0): each tap is the slow COST2100 magnitude times a
# Rician factor sqrt(K/(K+1)) + sqrt(1/(K+1)) * g_k(t), where g_k is a real,
# unit-variance Jakes (sum-of-sinusoids) process with normalized Doppler
# doppler = f_D * T_symbol. g_k runs continuously across all words of a
# repetition, so the taps change within a word, not just between words.
JAKES_SINUSOIDS = 16
FAST_FADING_SEED = 91138233


def jakes_process(n_samples: int, n_taps: int, doppler: float, rng: mtrand.RandomState) -> np.ndarray:
    """[n_samples, n_taps] independent real Jakes fading processes, unit variance."""
    t = np.arange(n_samples).reshape(-1, 1, 1)
    alpha = rng.uniform(0, 2 * np.pi, (1, n_taps, JAKES_SINUSOIDS))
    phi = rng.uniform(0, 2 * np.pi, (1, n_taps, JAKES_SINUSOIDS))
    return np.sqrt(2 / JAKES_SINUSOIDS) * np.cos(2 * np.pi * doppler * np.cos(alpha) * t + phi).sum(axis=2)


class ChannelModelDataset(Dataset):
    """
    Dataset object for the channel. Used in training and evaluation to draw minibatches of channel words and transmitted
    """

    def __init__(self, channel_type: str,
                 block_length: int,
                 transmission_length: int,
                 words: int,
                 memory_length: int,
                 channel_coefficients: str,
                 random: mtrand.RandomState,
                 word_rand_gen: mtrand.RandomState,
                 noisy_est_var: float,
                 fading_taps_type: int,
                 use_ecc: bool,
                 n_symbols: int,
                 fading_in_channel: bool,
                 fading_in_decoder: bool,
                 phase: str,
                 doppler: float = 0.0,
                 rician_k: float = 3.0):

        self.block_length = block_length
        self.transmission_length = transmission_length
        self.word_rand_gen = word_rand_gen if word_rand_gen else np.random.RandomState()
        self.random = random if random else np.random.RandomState()
        self.channel_type = channel_type
        self.words = words
        self.memory_length = memory_length
        self.channel_coefficients = channel_coefficients
        self.noisy_est_var = noisy_est_var
        self.fading_taps_type = fading_taps_type
        self.fading_in_channel = fading_in_channel
        self.fading_in_decoder = fading_in_decoder
        self.n_symbols = n_symbols
        self.phase = phase
        self.doppler = doppler
        self.rician_k = rician_k
        self.last_taps = None  # [words, T, L] true per-sample taps of the last fast-fading draw
        # Fast-fading draws are not cached; val draws are instead seeded per rep
        # (see get_snr_data), which keeps them paired across detectors.
        self.use_cache = doppler == 0
        if use_ecc:
            self.encoding = lambda b: encode(b, self.n_symbols)
        else:
            self.encoding = lambda b: b

    def get_snr_data(self, snr: float, gamma: float, database: list, rep: int = None):
        # if database is None:
        #     database = []
        word_rand_gen, noise_rand_gen = self.word_rand_gen, self.random
        fast = self.doppler > 0
        if fast:
            fading_rng = self.random
            if self.phase == 'val':
                # one seed per rep (a fixed one for rep=None, i.e. the training-time
                # validation set), shared by words, noise and fading: every detector
                # evaluated at rep r sees exactly the same transmission
                seed = FAST_FADING_SEED + (0 if rep is None else 1 + rep) * 7919 + int(round(10 * snr))
                word_rand_gen = noise_rand_gen = fading_rng = np.random.RandomState(seed)
            T = self.transmission_length
            fading = jakes_process(self.words * T, self.memory_length, self.doppler, fading_rng)
            k = self.rician_k
            fading = np.sqrt(k / (k + 1)) + np.sqrt(1 / (k + 1)) * fading
            taps = []
        b_full = np.empty((0, self.block_length))
        y_full = np.empty((0, self.transmission_length))
        if self.phase == 'val':
            index = 0
        else:
            index = 0  # random.randint(0, 1e6)
        # accumulate words until reaches desired number
        while y_full.shape[0] < self.words:
            # generate word
            b = word_rand_gen.randint(0, 2, size=(1, self.block_length))
            # encoding - errors correction Code
            c = self.encoding(b).reshape(1, -1)
            # add zero bits
            padded_c = np.concatenate([c, np.zeros([c.shape[0], self.memory_length])], axis=1)
            # transmit
            h = estimate_channel(self.memory_length, gamma,
                                 channel_coefficients=self.channel_coefficients,
                                 noisy_est_var=self.noisy_est_var,
                                 fading=self.fading_in_channel if self.phase == 'val' else self.fading_in_decoder,
                                 index=index,
                                 fading_taps_type=self.fading_taps_type)
            if fast:
                w = y_full.shape[0]
                h = h * fading[w * T:(w + 1) * T]  # [T, L] per-sample taps
                taps.append(h)
            y = self.transmit(padded_c, h, snr, noise_rand_gen)
            # accumulate
            b_full = np.concatenate((b_full, b), axis=0)
            y_full = np.concatenate((y_full, y), axis=0)
            index += 1

        database.append((b_full, y_full))
        if fast:
            self.last_taps = np.stack(taps)

    def transmit(self, c: np.ndarray, h: np.ndarray, snr: float, random: mtrand.RandomState = None):
        if self.channel_type == 'ISI_AWGN':
            # modulation
            s = BPSKModulator.modulate(c)
            # transmit through noisy channel
            y = ISIAWGNChannel.transmit(s=s, random=random if random is not None else self.random,
                                        h=h, snr=snr, memory_length=self.memory_length)
        else:
            raise Exception('No such channel defined!!!')
        return y

    def __getitem__(self, snr_list: List[float], gamma: float, rep: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get data for given SNRs and gamma. Uses cache if available.

        `rep`, when given, distinguishes independent repeated draws under
        otherwise-identical parameters -- e.g. Monte-Carlo evaluation
        repetitions, which are supposed to be independent trials. Without it,
        every call with the same (snr, gamma, phase, ...) hits the same cache
        entry and gets back byte-identical data every time, silently
        collapsing "N repetitions" into one repetition measured N times.
        Callers that want the old "one fixed draw, reused" behavior (e.g. a
        training set reused across minibatches) simply omit it.
        """

        # Check if we can use cache for all SNRs
        if self.use_cache and len(snr_list) == 1 and (rep is None or rep < CACHE_MAX_REP):
            snr = snr_list[0]
            # The fading flag actually used by get_snr_data depends on the phase
            fading = self.fading_in_channel if self.phase == 'val' else self.fading_in_decoder
            cache_params = dict(
                snr=snr, gamma=gamma,
                block_length=self.block_length,
                transmission_length=self.transmission_length,
                words=self.words,
                channel_coefficients=self.channel_coefficients,
                phase=self.phase,
                memory_length=self.memory_length,
                noisy_est_var=self.noisy_est_var,
                fading_taps_type=self.fading_taps_type,
                n_symbols=self.n_symbols,
                fading=fading,
                rep=rep
            )
            cache_filename = _data_cache.get_cache_filename(**cache_params)

            # Validate cache exists and matches parameters
            cache_valid = _data_cache.validate_cache(cache_filename, **cache_params)
            
            # Load from cache if valid
            if cache_valid:
                b, y = _data_cache.load_to_gpu_chunks(cache_filename, device)
                return b, y
            else:
                # Generate data (cache doesn't exist or parameters mismatch)
                print(f"[DataCache] Generating new data for SNR={snr}, gamma={gamma}, phase={self.phase}")
                database = []
                self.get_snr_data(snr, gamma, database)
                b, y = (np.concatenate(arrays) for arrays in zip(*database))
                
                # Save to cache with metadata
                _data_cache.save_to_cache(cache_filename, b, y, **cache_params)
                
                # Convert to GPU tensors
                b, y = torch.Tensor(b).to(device=device), torch.Tensor(y).to(device=device)
                return b, y
        
        # Fallback: original behavior for multiple SNRs or cache disabled
        database = []
        [self.get_snr_data(snr, gamma, database, rep) for snr in snr_list]
        b, y = (np.concatenate(arrays) for arrays in zip(*database))
        b, y = torch.Tensor(b).to(device=device), torch.Tensor(y).to(device=device)
        return b, y

    def __len__(self):
        return self.transmission_length
