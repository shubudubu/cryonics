# cryonics
Cryonics is a machine learning-based system designed to help parents and caregivers understand their baby's cries.  By analyzing the audio characteristics of a cry, Cryonics identifies the most likely underlying cause, such as hunger, discomfort, pain, burping, or tiredness.  The system uses advanced audio processing techniques to extract meaningful features like MFCCs, RMS, ZCR, and STFT, which are then fed into a machine learning model trained on a large dataset of labeled baby cries.  This allows Cryonics to accurately classify the reason behind the cry.

While currently in development for real-time integration, the project ultimately aims to provide instant cry analysis.  Cryonics is particularly beneficial for deaf parents, new parents, and caregivers who may struggle to interpret a baby's cries.  By providing quick and accurate information, Cryonics reduces parental stress and improves infant care.

Beyond cry analysis, Cryonics is also developing a community support feature, allowing parents to submit queries and receive expert advice on infant care.  Furthermore, the project is exploring the potential of similar audio analysis techniques for early autism detection, demonstrating its commitment to advancing infant well-being.

<b>Labeling;</b>
0: belly_pain
1: burping
2: discomfort
3: hungry
4: tired

<b>Feature Extraction;</b>

<b>Time-Domain Features:</b>
Amplitude_Envelope_Mean: The average amplitude of the cry signal over time. Higher values generally correspond to louder cries.
RMS_Mean (Root Mean Square Mean): Another measure of the average loudness or intensity of the cry. Similar to amplitude envelope, higher RMS values indicate a louder cry.
ZCR_Mean (Zero Crossing Rate Mean): The average number of times the signal crosses the zero-amplitude line per unit of time. A higher ZCR often indicates higher frequency components in the sound, which can be related to the pitch or sharpness of the cry.

<b>Frequency-Domain Features:</b>
STFT_Mean (Short-Time Fourier Transform Mean): The average magnitude of the STFT over time. The STFT breaks the audio into short segments and calculates the frequency spectrum within each segment. This feature captures the overall frequency content of the cry.
SC_Mean (Spectral Centroid Mean): The average frequency "center of gravity" of the cry's spectrum. It gives an idea of where the dominant frequencies lie. A higher spectral centroid suggests a higher-pitched cry.
SBAN_Mean (Spectral Bandwidth Mean): The average width of the frequency band occupied by the cry's sound. It provides information about the spread of frequencies present.
SCON_Mean (Spectral Contrast Mean): Measures the difference in energy between the peaks and valleys in the frequency spectrum. It helps to distinguish between different types of sounds.
MelSpec (Mel Spectrogram): A visual representation of the cry's frequencies over time, transformed using the Mel scale, which is closer to how humans perceive pitch. This feature captures the distribution of energy across different Mel frequency bands.

<b>MFCCs (Mel-Frequency Cepstral Coefficients):</b>
MFCCs1 - MFCCs13: These are 13 coefficients that represent the spectral envelope of the cry. They are widely used in audio analysis and are particularly good at capturing information about the vocal tract and the way sounds are produced. They are calculated using the Mel scale.
delMFCCs1 - delMFCCs13 (Delta MFCCs): These represent the rate of change of the MFCCs over time. They capture how the spectral envelope is changing, which can be important for distinguishing between different cry types.
del2MFCCs1 - del2MFCCs13 (Delta-Delta MFCCs): These represent the rate of change of the delta MFCCs, essentially capturing the acceleration of the spectral changes. They provide even more detailed information about the dynamics of the cry.
In summary, your feature set combines information about the loudness, pitch, frequency content, and how these characteristics change over time.  These features are then used to train your machine learning model to distinguish between the different cry types.  The MFCCs, along with their delta and delta-delta values, are particularly important and are often the most discriminative features in audio classification tasks.

![training](images/Screenshot%20(555).png)

![evaluation](images/Screenshot%20(556).png)

![prediction](images/Screenshot%20(557).png)

![prediction](images/Screenshot%20(558).png)
