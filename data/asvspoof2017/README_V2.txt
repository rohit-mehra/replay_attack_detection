――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
The 2nd Automatic Speaker Verification Spoofing
and Countermeasures Challenge (ASVspoof 2017) Database VERSION 2

********************************************************                                                     
IMPORTANT NOTICE

This is the VERSION 2 of the ASVspoof 2017 database. Please indicate explicitly the version number
of the database used in your papers. Refer to 'CHANGE_LOG_V2.txt' for further information about the changes introduced in this VERSION 2.
********************************************************

names: 
Tomi Kinnunen (1) 
Md Sahidullah (1) 
Héctor Delgado (2)
Massimiliano Todisco (2)
Nicholas Evans (2) 
Junichi Yamagishi (3),(4)
Kong Aik Lee (5)

affiliations: 
(1) University of Eastern Finland, Finland
(2) EURECOM, France
(3) University of Edinburgh, UK
(4) National Institute of Informatics, Japan
(5) Institute for Infocomm Research, Singapore

Copyright (c) 2018  
The Centre for Speech Technology Research (CSTR)
University of Edinburgh

――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

OVERVIEW
This is a database used for the Second Automatic Speaker Verification Spoofing and Countermeasuers Challenge, for short, ASVspoof 2017 (http://www.asvspoof.org) organized by Tomi Kinnunen, Md Sahidullah, Héctor Delgado, Massimiliano Todisco, Nicholas Evans, Junichi Yamagishi, Kong Aik Lee in 2017.

The ASVspoof challenge aims to encourage further progress through (i) the collection and distribution of a standard dataset with varying spoofing attacks implemented with multiple, diverse algorithms and (ii) a series of competitive evaluations for automatic speaker verification. 

The ASVspoof 2017 challenge follows on from two special sessions on spoofing and countermeasures for automatic speaker verification held during INTERSPEECH 2013 and 2015. While the first edition [1] in 2013 was targeted mainly at increasing awareness of the spoofing problem, the 2015 edition [2] included a first challenge on the topic, with commonly defined evaluation data, metrics and protocols. 

The task in ASVspoof 2015 was to discriminate genuine human speech from speech produced using text-to-speech (TTS) and voice conversion (VC) attacks [3]. The challenge was drawn upon state-of-the-art TTS and VC attacks data prepared for the “SAS” corpus [4] by TTS and VC researchers.

The primary technical goal of ASVspoof 2017 is to assess spoofing attack detection accuracy with ‘out in the wild’ conditions, thereby advancing research towards generalized spoofing countermeasure, in particular to detect replay [5]. In addition, ASVspoof 2017 attempts to better interlink the research efforts from spoofing and text-dependent ASV communities. To this end, ASVspoof 2017 makes an extensive use of the recent text-dependent RedDots corpus [6], as well as a replayed version of the same data [7].

The ASVspoof 2017 database contains large amount of speech data collected from 179 replay sessions in 61 unique replay configurations. Number of speakers is 42. A replay configuration means a unique combination of room, replay device and recording device, while a session refers to a set of source files, which share the same replay configuration. 

Below are some details about the database:

1. Training and development data are included in 'ASVspoof2017_V2_train.zip’ ’ASVspoof2017_V2_dev.zip’. Training dataset contains audio files with known ground-truth which can be used to train systems which can distinguish between genuine and spoofed speech. The development dataset contains audio files with known ground-truth which can be used for the development of spoofing detection algorithms.

2. Evaluation data is available in 'ASVspoof2017_V2_eval.zip’. 

3. Protocol and keys are available in 'protocol_V2.zip’.

4. Additional Instructions_V2.txt file is included in packages. There are originally used for the challenge participant to explain the database.

5. About how to compute the EERs, please refer the evaluation plan included in this repository.

6. To compare with the challenge results, please refer the summary paper of the challenge included in this repository.

7. The baseline results based on CQCC [8] can be reproduced using publicly released Matlab-based implementation of a replay attack spoofing detector http://www.asvspoof.org/data2017/baseline_CM.zip 

COPYING 
You are free to use this database under Creative Commons Attribution-NonCommercial License (CC-BY-NC). 

Regarding Creative Commons License: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0), please see 
https://creativecommons.org/licenses/by-nc/4.0/

THIS DATABASE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS DATABASE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


ACKNOWLEDGEMENTS
This research was partly funded by the OCTAVE Project (#647850), funded by the Research European Agency (REA) of the European Commission, in its framework programme Horizon 2020, by the Academy of Finland (grant #288558) and by MEXT KAKENHI (Grant Numbers 26280066, 15H01686, 16H06302).


REFERENCES
[1] Nicholas Evans, Tomi Kinnunen and Junichi Yamagishi, "Spoofing and countermeasures for automatic speaker verification", Interspeech 2013,  925-929, August 2013

[2] Zhizheng Wu, Tomi Kinnunen, Nicholas Evans, Junichi Yamagishi, Cemal Hanilc, Md Sahidullah Aleksandr Sizov, "ASVspoof 2015: the First Automatic Speaker Verification Spoofing and Countermeasures Challenge", Proc. Interspeech 2015  2037-2041 September 2015

[3] Zhizheng Wu, Junichi Yamagishi, Tomi Kinnunen, Cemal Hanilci, Md Sahidullah, Aleksandr Sizov, Nicholas Evans, Massimiliano Todisco, Hector Delgado, "ASVspoof: the Automatic Speaker Verification Spoofing and Countermeasures Challenge", Special Issue on Spoofing and Countermeasures for Automatic Speaker Verification, IEEE Journal of Selected Topics in Signal Processing 11(4), 588-604, June 2017

[4] Zhizheng Wu, Phillip L. De Leon, Cenk Demiroglu, Ali Khodabakhsh, Simon King, Zhen-Hua Ling, Daisuke Saito, Bryan Stewart, Tomoki Toda, Mirjam Wester, and Junichi Yamagishi, "Anti-Spoofing for Text-Independent Speaker Verification: An Initial Database, Comparison of Countermeasures, and Human Performance", IEEE/ACM Transactions on Audio, Speech, and Language Processing, 24(4), 768-783, April 2016

[5] Tomi Kinnunen, Md Sahidullah, Héctor Delgado, Massimiliano Todisco, Nicholas Evans, Junichi Yamagishi, Kong Aik Lee, "The ASVspoof 2017 Challenge: Assessing the Limits of Replay Spoofing Attack Detection", Proc. Interspeech 2017, August 2017

[6] K. Lee, A. Larcher, G. Wang, P. Kenny, N. Brümmer, D. A. van Leeuwen, H. Aronowitz, M. Kockmann, C. Vaquero, B. Ma, H. Li, T. Stafylakis, M. J. Alam, A. Swart, and J. Perez, “The RedDots data collection for speaker recognition,” in Proc. INTER- SPEECH, 2015, pp. 2996–3000.

[7] T. Kinnunen, M. Sahidullah, M. Falcone, L. Costantini, R. G. Hautamäki, D. Thomsen, A. Sarkar, Z.-H. Tan, H. Delgado, M. Todisco, N. Evans, V. Hautamäki, and K. A. Lee, “Reddots replayed: A new replay spoofing attack corpus for text-dependent speaker verification research,” in Proc. ICASSP, New Orleans, USA, 2017.

[8] Massimiliano Todisco, Héctor Delgado, Nicholas Evans, “Constant Q cepstral coefficients: A spoofing countermeasure for automatic speaker verification”, Computer Speech & Language, Pages 516-535, Volume 45, September 2017, 

POINTERS 
ASVSpoof challenge: http://www.asvspoof.org
ASVSpoof evaluation plan: http://www.asvspoof.org/data2017/asvspoof-2017_evalplan_v1.1.pdf
ASVSpoof summary paper: http://www.asvspoof.org/asvspoof2017overview_cameraReady.pdf
SAS corpus: http://dx.doi.org/10.7488/ds/252
VCTK corpus: http://dx.doi.org/10.7488/ds/1994
ASVspoof 2015 corpus: http://dx.doi.org/10.7488/ds/298
RedDots corpus: https://sites.google.com/site/thereddotsproject/ 

