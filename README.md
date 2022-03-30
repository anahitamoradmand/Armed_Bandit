# RL

2-armed Bandit problem:

Think of an agent that plays a 2-armed bandit, trying to maximize its total reward. In each step, the 
# agent selects one of the levers and is given some reward according to the reward distribution of that 
# lever. Assume that reward distribution for the first lever is a Gaussian with 𝜇1 = 5, 𝜎12 = 10, and for 
# the second lever is a binomial Gaussian with 𝜇21 = 10, 𝜎21^2 = 15, 𝜇22 = 4, 𝜎22^2 = 10, which means 
# that  the  resulting  output  will  be  uniformly  probable  from  these  two  Gaussian  distributions. 
# In  this  problem,  we  assume  the  reward  distributions  are  unknown,  and  the  agent  only  
# sees  a  realization  of  reward  after  selecting  an  action.  The  agent  takes  action  according  to  the  𝜖𝜖-
# greedy action selection policy with parameter 𝜖. We  consider  the  agent  selects  1000  actions,  which  is  referred  to  as  step/time.  In  order  to  have  
# smooth results, we repeat 1000 steps for 100 independent runs. 

# Part a - In this part, set the initial Q values at the beginning of each run as 𝑄(𝑎1) = 𝑄(𝑎2) = 0. Assuming 
# action 𝑎 is  selected  at  time  step  𝑘 and  the  reward  𝑟_𝑘  is  observed,  the  Q-value  for  the  
# corresponding  action  will  be  updated  according  to:  𝑄(𝑎) = 𝑄(𝑎) + 𝛼( r − 𝑄(𝑎) ).  For  the  
# learning rates, consider the following values: 𝛼 = 1, 𝛼 = 0.9𝑘 , 𝛼 =1/1+Ln(1+𝑘) and 𝛼 = 1/𝑘 and for 
# the  𝜖-greedy  policy,  use  𝜖 = 0, 0.1, 0.2, 0.5.  
# For the 𝑖𝑖th independent run, you need to keep track of accumulated rewards as: 
# Acc R^i_k=(1/k) sum_{j=1 to k} r_j
# where Acc R^i_k denotes the average reward per step obtained by the agent up to the time step 𝑘 in  the  ith  independent  run.  Then  the  average  over  100  independent  runs  of  accumulated 
# rewards Acc R^i_k can be obtained at any given step/time 𝑘 = 1, ... ,1000 as: 
# Acc R_k=(1/100) sum_{j=1 to 100} Acc R^i_k

# Part b - For  a  fixed  𝛼 = 0.1  and  𝜖= 0.1,  use  the  following  optimistic  initial  values  and  compare  the  
# results:  𝑄 = [0 0],  𝑄  = [5 7], 𝑄 = [20 20]  (note  that  𝑄 = [𝑄(𝑎1)  𝑄(𝑎2)]).

# Part c - For a fixed 𝛼 = 0.1, use the Gradient-Bandit policy with 𝐻1(𝑎1) = 𝐻1(𝑎2) = 0. Plot the average 
# accumulated  reward  with  respect  to  step/time.  How  the  results  are  different  from  𝜖-greedy 
#results with 𝑄(𝑎1) = 𝑄(𝑎2) = 0, 𝛼𝛼 = 0.1 and 𝜖 = 0.1? 
