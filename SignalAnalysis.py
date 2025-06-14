import pandas as pd
import numpy as np
import json
from io import StringIO
from scipy import signal, fft
from scipy.signal import decimate, butter, filtfilt, medfilt, get_window

class SignalAnalysis:
    def __init__(self):
        self.df = pd.DataFrame()
        self.processed_df = pd.DataFrame()
        self.damped_df = pd.DataFrame()
        self.fft_df = pd.DataFrame()
        self.autocorr_df = pd.DataFrame()
        self.crosscorr_df = pd.DataFrame()

    def load_data(self, file):
        try:
            self.df = pd.read_csv(file, header=0)
            self.processed_df = self.df.copy()
            print(f"File loaded successfully. DataFrame shape: {self.df.shape}")
            # print(self.df.head()) # print first few rows for verification
            # print(self.df.shape)  # print shape of the DataFrame
        except Exception as e:
            raise ValueError(f"Error reading file: {e}")

    def detrend_data(self):
        """
        Detrend signal data by removing linear trends while preserving time column.
        
        Raises:
            ValueError: If DataFrame is empty or contains no signal columns
        """
        if self.df.empty:
            raise ValueError("DataFrame is empty. Load data first.")
        
        if self.df.shape[1] < 2:
            raise ValueError("DataFrame must contain at least one signal column besides time")
        
        # Exclude 'time' column from detrending
        time = self.df.iloc[:, 0]
        signal_data = self.df.iloc[:, 1:]

        try:
            # Apply detrending
            detrended_data = signal.detrend(signal_data, axis=0)

            # Reconstruct DataFrame with time and detrended signals
            self.detrended_df = pd.DataFrame(detrended_data, columns=signal_data.columns)
            self.detrended_df.insert(0, 'time', time)
            self.processed_df = self.detrended_df.copy()
            # print(self.processed_df.head())  # print first few rows for verification
            print(f"Detrending complete: new shape: {self.processed_df.shape}")
            
        except Exception as e:
            raise ValueError(f"Error during detrending: {str(e)}")

    def decimate_data(self):
        factor = 2  # Default decimation factor
        if self.processed_df.empty:
            raise ValueError("No data loaded in df.")
        if self.processed_df.shape[1] < 11:
            raise ValueError("DataFrame must contain at least 11 signal columns to decimate.")
        if factor <= 0:
            raise ValueError("Decimation factor must be a positive integer.")
        
        # Decimate the data by selecting every nth row
        self.processed_df = self.processed_df.iloc[::factor].reset_index(drop=True)
        # print(self.processed_df.head())  # print first few rows for verification
        print(f"Decimation complete: new shape: {self.processed_df.shape}")


    def apply_filter(self, filter: dict):
        """
        Apply a specified filter to the processed DataFrame.

        Args:
            filter (dict): Dictionary with structure:
                {
                    "filter_type": "<filter_name>",
                    "params": { ... }
                }

        Supported filters:
            - low_pass: needs 'cutoff_freq', 'filter_order', 'fs'
            - high_pass: needs 'cutoff_freq', 'filter_order', 'fs'
            - band_pass: needs 'low_cutoff_freq', 'high_cutoff_freq', 'filter_order', 'fs'
            - band_stop: needs 'start', 'end', 'filter_order', 'fs'
            - moving_avg: needs 'window_size'
            - median_filter: needs 'window_size'

        Updates:
            self.processed_df: filtered DataFrame
        """
        if self.processed_df is None or self.processed_df.empty:
            raise ValueError("No data to filter. Please load and preprocess the data first.")

        filter_type = filter.get("filter_type")
        params = filter.get("params", {})

        if not filter_type:
            raise ValueError("Missing filter_type in filter dictionary.")

        fs = params.get("fs", 1.0)  # Default sampling frequency

        data = self.processed_df.copy()
        filtered = pd.DataFrame(index=data.index)

        for col in data.columns:
            if col.lower() == 'time' or not np.issubdtype(data[col].dtype, np.number):
                filtered[col] = data[col]  # Pass non-numeric columns untouched
                continue

            signal = data[col].values

            try:
                if filter_type == "low_pass":
                    b, a = butter(params["filter_order"], params["cutoff_freq"] / (0.5 * fs), btype="low")
                    filtered[col] = filtfilt(b, a, signal)

                elif filter_type == "high_pass":
                    b, a = butter(params["filter_order"], params["cutoff_freq"] / (0.5 * fs), btype="high")
                    filtered[col] = filtfilt(b, a, signal)

                elif filter_type == "band_pass":
                    low = params["low_cutoff_freq"] / (0.5 * fs)
                    high = params["high_cutoff_freq"] / (0.5 * fs)
                    b, a = butter(params["filter_order"], [low, high], btype="band")
                    filtered[col] = filtfilt(b, a, signal)

                elif filter_type == "band_stop":
                    start = params["start"] / (0.5 * fs)
                    end = params["end"] / (0.5 * fs)
                    b, a = butter(params["filter_order"], [start, end], btype="bandstop")
                    filtered[col] = filtfilt(b, a, signal)

                elif filter_type == "moving_avg":
                    window = params.get("window_size", 5)
                    filtered[col] = pd.Series(signal).rolling(window, center=True, min_periods=1).mean()

                elif filter_type == "median_filter":
                    window = params.get("window_size", 5)
                    if window % 2 == 0:
                        window += 1  # Must be odd
                    filtered[col] = medfilt(signal, kernel_size=window)

                else:
                    raise ValueError(f"Unsupported filter type: {filter_type}")

            except Exception as e:
                raise RuntimeError(f"Error applying filter on column '{col}': {e}")

        self.processed_df = filtered
        print(self.processed_df.head())
        print(f"Filter '{filter_type}' applied successfully. Shape: {self.processed_df.shape}")

    # def apply_filter(self, filter: dict):
    #     """
    #     Apply a specified filter to the processed DataFrame.
    
    #     Args:
    #         filter (dict): Dictionary with structure:
    #             {
    #                 "filter_type": "<filter_name>",
    #                 "params": { ... }
    #             }
    
    #     Supported filters:
    #         - low_pass
    #         - high_pass
    #         - band_pass
    #         - band_stop
    #         - moving_avg
    #         - median_filter
    
    #     Notes:
    #         `params` must include necessary values such as:
    #             - cutoff_freq, filter_order, fs (sampling freq) for low/high-pass
    #             - low_cutoff_freq, high_cutoff_freq, filter_order for band-pass
    #             - window_size for moving_avg or median_filter
    
    #     Updates:
    #         self.processed_df: filtered DataFrame
    #     """
    #     if self.processed_df is None or self.processed_df.empty:
    #         raise ValueError("No data to filter. Please load and preprocess the data first.")

    #     filter_name = filter.get("filter_type")
    #     params = filter.get("params", {})
    #     fs = params.get("fs", 1.0)  # Default to 1 Hz if not provided
    
    #     data = self.processed_df.copy()
    #     filtered = pd.DataFrame(index=data.index)
    
    #     for col in data.columns:
    #         signal = data[col].values
    
    #         if filter_name == "low_pass":
    #             b, a = butter(params["filter_order"], params["cutoff_freq"] / (0.5 * fs), btype="low")
    #             filtered[col] = filtfilt(b, a, signal)
    
    #         elif filter_name == "high_pass":
    #             b, a = butter(params["filter_order"], params["cutoff_freq"] / (0.5 * fs), btype="high")
    #             filtered[col] = filtfilt(b, a, signal)
    
    #         elif filter_name == "band_pass":
    #             low = params["low_cutoff_freq"] / (0.5 * fs)
    #             high = params["high_cutoff_freq"] / (0.5 * fs)
    #             b, a = butter(params["filter_order"], [low, high], btype="band")
    #             filtered[col] = filtfilt(b, a, signal)
    
    #         elif filter_name == "band_stop":
    #             start = params["start"] / (0.5 * fs)
    #             end = params["end"] / (0.5 * fs)
    #             b, a = butter(params["filter_order"], [start, end], btype="bandstop")
    #             filtered[col] = filtfilt(b, a, signal)
    
    #         elif filter_name == "moving_avg":
    #             window = params["window_size"]
    #             filtered[col] = pd.Series(signal).rolling(window, center=True, min_periods=1).mean()
    
    #         elif filter_name == "median_filter":
    #             window = params["window_size"]
    #             filtered[col] = medfilt(signal, kernel_size=window)
    
    #         else:
    #             raise ValueError(f"Unknown filter: {filter_name}")
    
    #     self.processed_df = filtered
    #     print(self.processed_df.head())
    #     print(f"Filter '{filter_name}' applied successfully. New shape: {self.processed_df.shape}")

    

    def fft(self, fft_params: dict):
        """
        Apply FFT on the processed data using given parameters.

        Args:
            fft_params (dict): Dictionary with keys:
                - fft_size (int): Number of FFT points.
                - sampling_rate (float): Sampling rate of the signal.
                - window (str): Type of window function to apply. One of:
                    ['rectangular', 'hanning', 'gaussian', 'blackman', 'kaiser', 'flattop']

        Stores:
            self.fft_df: DataFrame with FFT results (magnitude spectrum).
        """
        if self.processed_df is None or self.processed_df.empty:
            raise ValueError("No processed data to perform FFT on.")

        fft_size = fft_params.get("fft_size", 1024)
        sampling_rate = fft_params.get("sampling_rate", 1.0)
        window_type = fft_params.get("window", "rectangular")

        window_map = {
            "rectangular": ("boxcar", None),
            "hanning": ("hann", None),
            "gaussian": ("gaussian", 7),  # Use default std
            "blackman": ("blackman", None),
            "kaiser": ("kaiser", 14),     # Use default beta
            "flattop": ("flattop", None)
        }

        if window_type not in window_map:
            raise ValueError(f"Unsupported window type: {window_type}")

        window_name, param = window_map[window_type]
        window = get_window((window_name, param), fft_size) if param else get_window(window_name, fft_size)

        freqs = np.fft.rfftfreq(fft_size, d=1.0 / sampling_rate)
        fft_result = pd.DataFrame(index=freqs)

        for col in self.processed_df.columns:
            if col.lower() == 'time':
                continue  # skip time column

            signal = self.processed_df[col].values[:fft_size]

            if len(signal) < fft_size:
                signal = np.pad(signal, (0, fft_size - len(signal)))

            windowed_signal = signal * window
            spectrum = np.fft.rfft(windowed_signal)
            magnitude = np.abs(spectrum)

            fft_result[col] = magnitude

        self.fft_df = fft_result
        # print(self.fft_df.head())  # ✅ correct attribute
        print(f"FFT applied successfully. Resulting shape: {self.fft_df.shape}")


    def damping(self, param: str):
        """
        Apply damping to the processed data.

        Args:
            param (str): Type of damping to apply. One of ['zero', 'half', 'full'].
                - 'zero' = no damping (original signal)
                - 'half' = signal amplitude is halved
                - 'full' = signal is suppressed to near-zero (simulated complete damping)

        Stores:
            self.damped_df: DataFrame containing the damped signal
        """
        if self.processed_df is None or self.processed_df.empty:
            raise ValueError("No processed data available for damping.")

        if param == 'zero':
            self.damped_df = self.processed_df.copy()
            
            print("Zero damping applied (no change to signal).")
        elif param == 'half':
            self.damped_df = self.processed_df * 0.5
            # print(self.damped_df.head())  # print first few rows for verification
            print("Half damping applied (signal amplitude halved).")
        elif param == 'full':
            self.damped_df = self.processed_df * 0.01  # Not exactly zero to preserve structure
            # print(self.damped_df.head())  # print first few rows for verification
            print("Full damping applied (signal suppressed to near-zero).")
        else:
            raise ValueError(f"Unsupported damping level: {param}. Choose from ['zero', 'half', 'full'].")
        

    def autocorrelate(self, max_lags=None):
        """
        Compute and normalize autocorrelation for each numeric column in processed_df.

        Args:
            max_lags (int, optional): Max number of lags to compute. Defaults to full length of signal.

        Stores:
            self.autocorr_df: DataFrame with autocorrelation for each column.
        """
        if self.processed_df is None or self.processed_df.empty:
            raise ValueError("No processed data available for autocorrelation.")

        autocorr_results = {}
        signal_length = len(self.processed_df)

        for col in self.processed_df.columns:
            if col.lower() == "time" or not np.issubdtype(self.processed_df[col].dtype, np.number):
                continue  # Skip non-numeric or time columns

            signal = self.processed_df[col] - self.processed_df[col].mean()
            full_autocorr = np.correlate(signal, signal, mode='full')
            full_autocorr = full_autocorr[full_autocorr.size // 2:]  # keep non-negative lags
            full_autocorr /= full_autocorr[0]  # normalize

            if max_lags is not None:
                autocorr_results[col] = full_autocorr[:max_lags]
            else:
                autocorr_results[col] = full_autocorr

        self.autocorr_df = pd.DataFrame(autocorr_results)
        self.autocorr_df.reset_index(inplace=True)
        self.autocorr_df.rename(columns={'index': 'lag'}, inplace=True)
        # print(self.autocorr_df.head())  # print first few rows for verification
        print("Autocorrelation computed successfully. Shape:", self.autocorr_df.shape)


    def cross_correlate(self, max_lags=None):
        """
        Compute normalized cross-correlation between all pairs of numeric columns in processed_df.

        Args:
            max_lags (int, optional): Max number of lags (both directions) to include. Defaults to full length.

        Stores:
            self.crosscorr_dict: Dictionary where keys are (col1, col2) tuples,
                                 and values are DataFrames with 'lag' and 'crosscorr' columns.
        """
        if self.processed_df is None or self.processed_df.empty:
            raise ValueError("No processed data available for cross-correlation.")

        numeric_cols = [col for col in self.processed_df.columns
                        if np.issubdtype(self.processed_df[col].dtype, np.number)
                        and col.lower() != 'time']

        self.crosscorr_dict = {}

        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col1 = numeric_cols[i]
                col2 = numeric_cols[j]

                x = self.processed_df[col1] - self.processed_df[col1].mean()
                y = self.processed_df[col2] - self.processed_df[col2].mean()

                corr = np.correlate(x, y, mode='full')
                norm_factor = np.std(x) * np.std(y) * len(x)
                corr /= norm_factor

                lags = np.arange(-len(x) + 1, len(x))

                if max_lags:
                    mid = len(corr) // 2
                    half = max_lags
                    corr = corr[mid - half: mid + half + 1]
                    lags = lags[mid - half: mid + half + 1]

                self.crosscorr_dict[(col1, col2)] = pd.DataFrame({
                    'lag': lags,
                    'crosscorr': corr
                })
        # print(f"Cross-correlation computed for {len(self.crosscorr_dict)} pairs of columns.")
        print("Cross-correlation computed successfully. Total pairs:", len(self.crosscorr_dict))



