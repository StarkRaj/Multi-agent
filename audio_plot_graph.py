import os
import numpy as np
import matplotlib.pyplot as plt
from moviepy import VideoFileClip
from scipy.io import wavfile

def extract_audio_from_video(video_path, audio_path):
    """
    Extract the audio track from a video file and save it as a separate audio file.

    Parameters:
        video_path (str): Path to the input video file.
        audio_path (str): Path where the extracted audio file will be saved.

    Raises:
        ValueError: If the video file does not contain an audio track.
    """
    clip = VideoFileClip(video_path)
    if clip.audio is not None:
        clip.audio.write_audiofile(audio_path, logger=None)
    else:
        raise ValueError("No audio track found in the video file.")


def read_audio(audio_path):
    """
    Read an audio file and return the sample rate and audio data.

    Parameters:
        audio_path (str): Path to the audio file.

    Returns:
        sr (int): Sample rate of the audio file.
        audio (np.ndarray): Audio data as a numpy array. If the audio is stereo,
        it is converted to mono by averaging the channels.
    """
    sr, audio = wavfile.read(audio_path)
    if len(audio.shape) == 2:
        audio = audio.mean(axis=1)
    return sr, audio


def compute_loudness(audio, sr, frame_duration=1.0):
    """
    Compute the loudness of the audio in dB over time.

    Parameters:
        audio (np.ndarray): Audio data as a numpy array.
        sr (int): Sample rate of the audio file.
        frame_duration (float): Duration of each frame in seconds.

    Returns:
        times (np.ndarray): Array of time values corresponding to each frame.
        loudness_db (np.ndarray): Array of loudness values in dB for each frame.
    """
    try:
        frame_size = int(sr * frame_duration)
        n_frames = int(np.ceil(len(audio) / frame_size))
        loudness_db = []
        times = []
        for i in range(n_frames):
            start = i * frame_size
            end = min((i + 1) * frame_size, len(audio))
            frame = audio[start:end]
            if len(frame) == 0:
                continue

            rms = np.sqrt(np.mean(frame.astype(float) ** 2))
            if rms < 1e-10:
                db = -100
            else:
                db = 20 * np.log10(rms / (2**15))
            loudness_db.append(db)
            times.append(start / sr)
        return np.array(times), np.array(loudness_db)
    except Exception as err:
        raise RuntimeError(f"Error in compute_loudness: {err}")


def moving_average(data, window_size):
    """
    Compute the moving average of the data.

    Parameters:
        data (np.ndarray): Input data array.
        window_size (int): Size of the moving average window.

    Returns:
        np.ndarray: Moving average of the input data.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')


def highlight_audio_section(ax, times, loudness, start_time, end_time, threshold, color, alpha=0.3, above=False):
    """ 
    Highlight a specific audio section on the graph.

    Parameters:
        ax (matplotlib.axes.Axes): The axes object to plot on.
        times (np.ndarray): Array of time values corresponding to each frame.
        loudness (np.ndarray): Array of loudness values in dB for each frame.
        start_time (float): Start time of the audio section.
        end_time (float): End time of the audio section.
        threshold (float): Threshold value for highlighting the audio section.
        color (str): Color of the highlighted audio section.
        alpha (float): Transparency of the highlighted audio section.
        above (bool): Whether to highlight the audio section above or below the threshold.
    """
    times_array = np.array(times)
    loudness_array = np.array(loudness)
    try:
        if above:
            condition = ((times_array >= start_time) & (times_array <= end_time) & (loudness_array > threshold))
            ax.fill_between(
                times, loudness_array, threshold, where=condition, color=color, alpha=alpha
            )
        else:
            condition = ((times_array >= start_time) & (times_array <= end_time) & (loudness_array < threshold))
            ax.fill_between(
                times, loudness_array, threshold, where=condition, color=color, alpha=alpha
            )
    except Exception as err:
        raise RuntimeError(f"Error highlighting audio section from {start_time} to {end_time}: {err}")


def plot_loudness(times, loudness, save_path, highlights=None):
    """
    Plot the loudness of the audio over time.

    Parameters:
        times (np.ndarray): Array of time values corresponding to each frame.
        loudness (np.ndarray): Array of loudness values in dB for each frame.
        save_path (str): Path to save the generated graph.
        highlights (list): List of dictionaries containing highlight parameters.
    """
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(times, loudness, color='#1f77b4', linewidth=2.5, label='Loudness in dB')
        # Reference lines
        ax.axhline(y=-22, color='red', linestyle='--', linewidth=2, alpha=0.8, label='High Audio -22 dB')
        ax.axhline(y=-26, color='blue', linestyle='--', linewidth=2, alpha=0.8, label='Low Audio -26 dB')

        if highlights:
            for highlight in highlights:
                try:
                    highlight_audio_section(ax, times, loudness, **highlight)
                except Exception as err:
                    print(f"Warning: Failed to highlight section {highlight}: {err}")

        # Add permanent legend entries for detected low/high audio
        low_patch = plt.Line2D([0], [0], color='blue', alpha=0.3, linewidth=8, label='Detected low audio for >= 5s')
        high_patch = plt.Line2D([0], [0], color='red', alpha=0.3, linewidth=8, label='Detected high audio for >= 5s')
        no_audio_patch = plt.Line2D([0], [0], color='gray', alpha=0.3, linewidth=8, label='Detected no audio for >= 5s')
        ax.add_line(low_patch)
        ax.add_line(high_patch)
        ax.add_line(no_audio_patch)

        ax.set_xlabel('Time (s)', fontsize=14)
        ax.set_ylabel('Loudness (dB)', fontsize=14)
        ax.set_title('Audio Loudness Over Time', fontsize=18, weight='bold')
        ax.set_yticks(np.arange(-60, 5, 5))
        ax.set_ylim(-60, 0)
        ax.set_xticks(np.arange(0, times[-1] + 5, 5))
        ax.set_xlim(0, times[-1])
        ax.tick_params(axis='both', which='major', labelsize=12)
        # Deduplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        unique = dict()
        for h, l in zip(handles, labels):
            if l not in unique and l != "":
                unique[l] = h
        ax.legend(unique.values(), unique.keys(), fontsize=13)
        ax.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
        plt.tight_layout()
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            except Exception as err:
                print(f"Error saving figure to {save_path}: {err}")
            finally:
                plt.close()
    except Exception as err:
        raise RuntimeError(f"Error occurred while plotting loudness: {err}")


def main(video_path: str, output_graph_path: str, graph_name: str, highlights: list = []):
    AUDIO_TEMP_PATH = 'temp_audio.wav'
    FRAME_DURATION = 0.1  # seconds
    smooth_window = 3
    save_path = os.path.join(output_graph_path, graph_name)
    try:
        extract_audio_from_video(video_path, AUDIO_TEMP_PATH)
        sr, audio = read_audio(AUDIO_TEMP_PATH)
        times, loudness_db = compute_loudness(audio, sr, frame_duration=FRAME_DURATION)
        loudness_db_smooth = moving_average(loudness_db, smooth_window)
        plot_loudness(times, loudness_db_smooth, save_path, highlights=highlights)
    except Exception as err:
        print(f"Error in main: {err}")
    finally:
        if os.path.exists(AUDIO_TEMP_PATH):
            try:
                os.remove(AUDIO_TEMP_PATH)
            except Exception as err:
                print(f"Error removing temp audio file: {err}")

if __name__ == "__main__":
    VIDEO_PATH = 'D:\\Projects\\Fun\\Audio_graph_plot\\assets\\audio_anomaly_all3.mp4'
    OUTPUT_GRAPH = 'D:\\Projects\\Fun\\Audio_graph_plot'
    LOUDNESS_HIGHLIGHTS = []
    LOUDNESS_HIGHLIGHTS = [
        {"start_time": 35, "end_time": 55, "threshold": -26, "color": 'blue', "alpha": 0.3, "above": False},
        {"start_time": 15, "end_time": 30, "threshold": -22, "color": 'red', "alpha": 0.3, "above": True},
        {"start_time": 6, "end_time": 12, "threshold": -26, "color": 'gray', "alpha": 0.3, "above": False}
    ]
    try:
        main(VIDEO_PATH, OUTPUT_GRAPH, f'{VIDEO_PATH.split("\\")[-1].split(".")[0]}_graph.png', LOUDNESS_HIGHLIGHTS)
    except Exception as err:
        print(f"Fatal error: {err}")
