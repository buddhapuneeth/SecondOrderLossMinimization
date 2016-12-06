We implemented 2 machines:
Logit.py - for NLL related experiments
Logit1.py - for HL related experiments

The code is modularized to run whatever experiments you want to run

Getting input data:
All the input data, is feeded into X vector and output labels to Y vector.
The input here is feeded from make_blobs with various parameters based on the experiment
For ex:
  In the code, you can see 
    data1- less scattered, clearly sepearable
    data2- overlapped data
  We need to uncomment whichever is needed

To have noise on data, you can see 2 ways of creating noise:
	1. To add error at the end - This creates noise in last 100 inputs.
	2. To add noise in intermidiate data - This uniformly distributes error over in 10% of the data

You need to uncomment corresponding module. We can have both or anyone or none.

To increase input params, we need to uncomment module MultiFeauture set.
	This code is to increase feature set from 2 to 5: X, Y, X*X, X*Y, Y*Y
	Accordingly we need to initialize weights in main code: 'betas' is the vector with weights

MAIN MODULE:
4 * 2 different algorithms are defined here : Pure NLL, NLL + Adagrad, NLL + RMSProp, NLL+ ADAM (in Logit.py)
										Pure HL, HL + Adagrad, HL + RMSProp, HL+ ADAM(in Logit1.py)

Ex: for running in NLL+ Adagrad, uncomment 
	#fitted_values, cost_iter = grad_desc_adagrad(betas, X, Y)
	Note: make sure you commented any of the 3 optimizer calls

For L2 on above 8 experiments (8*2)
You can define 'alpha' which is like lamda in L2. It is defined at class-level
Ex: for running NLL+Adagrad with L2, just change grad_desc_adagrad(betas, X, Y) to grad_desc_adagrad(betas, X, Y, L2 = True)

We executed more than 50 experiments to observe various behaviours. The results are documented CASE wise in Report.txt and you can find cost function for particular case in screenshots directory using caseid, which you can get from Report.txt