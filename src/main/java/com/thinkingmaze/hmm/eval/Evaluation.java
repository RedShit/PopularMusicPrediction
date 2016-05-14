package com.thinkingmaze.hmm.eval;

import com.thinkingmaze.hmm.HMM;
import com.thinkingmaze.hmm.observation.Observation;

public abstract class Evaluation {
	HMM hmm;
	int stateLength;
	int obsLength;
	int deltaLength;
	
	public abstract double alpha(int obsIndex, int stateIndex);
	
	public abstract double forward();

	public abstract double epsilon(int obsIndex, int deltaIndex);

}
