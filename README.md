# DQN
This is DQN implemented with tensorflow 2.
DQN implemented in two version. The original DQN (Nature 2015) and multi-step DQN.
The experiment has done under the OpenAI gym cartpole-v1 environment.
## Requirements
* Python 3.8.2
* tensorflow 2.2.0rc2
* matplotlib 3.2.1
## Usage
Training only original DQN.
<pre>
<code>
python script.py orgDQN
</code>
</pre>
Training original DQN and multi-step DQN in a sequence.
<pre>
<code>
python script.py orgDQN multistep
</code>
</pre>
## Result
If the average step count exceed 475, the training ends early. 
<img src="./final result(org: 488 epi, 475.39 step| mult: 303 epi, 476.4 step .png" width="450px" height="300px"></img><br/>
|model|end episode|final average step|
|------|---|---|
|Original DQN|488|475.39|
|Multi-step DQN (n=3)|303|476.4|

