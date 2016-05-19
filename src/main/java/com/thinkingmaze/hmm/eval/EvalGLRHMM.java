package com.thinkingmaze.hmm.eval;

import java.util.Arrays;

import com.thinkingmaze.hmm.HMM;
import com.thinkingmaze.hmm.observation.Observation;
import com.thinkingmaze.hmm.train.TrainGLRHMM;
import com.thinkingmaze.hmm.train.Training;

public class EvalGLRHMM extends Evaluation {

	private double[][] tAlpha;
	
	public void trainPiMatrix(int obvValue) {
		Observation observation = new Observation(new int[]{obvValue});
		Training trainPiHMM = new TrainGLRHMM(observation, super.hmm);
		for(int e = 0; e < 20; e++){
			trainPiHMM.trainPiMatrix();
		}
	}
	
	public EvalGLRHMM(HMM hmm) {
		super.hmm = hmm;
		super.stateLength = hmm.getStateLength();
		super.deltaLength = hmm.getDeltaLength();
	}
	
	private double getAlpha(int obsIndex, int stateIndex){
		if(this.tAlpha[obsIndex-1][stateIndex-1]>=0)
			return this.tAlpha[obsIndex-1][stateIndex-1];
		if (obsIndex == 1) {
			return this.tAlpha[obsIndex-1][stateIndex-1]=hmm.getPiMatrix(stateIndex);
		}
		
		double sum = 0;

		for (int j = 1; j <= stateLength; j++) {
			System.out.println(j + " " + stateIndex + " " + hmm);
			sum += hmm.getStateTransitProbMatrix(j, stateIndex)
					* getAlpha(obsIndex - 1 , j);
		}
		return this.tAlpha[obsIndex-1][stateIndex-1]=sum;
	}

	private double alpha(int obsIndex, int stateIndex) {
		if(stateIndex > super.stateLength)
			throw new RuntimeException("stateIndex > super.stateLength");
		
		return getAlpha(obsIndex, stateIndex);
	}
	public double epsilon(int obsIndex, int deltaIndex) {
		if(deltaIndex > super.deltaLength)
			throw new RuntimeException("deltaIndex > super.deltaLength");
		this.tAlpha = new double[obsIndex][];
		for(int i = 0; i < obsIndex; i++){
			this.tAlpha[i] = new double[super.stateLength];
			Arrays.fill(this.tAlpha[i], -1.0);
		}
		double res = 0.0;
		for(int i = 1; i <= super.stateLength; i++){
			res += alpha(obsIndex,i)*hmm.getEmissionProbMatrix(i, deltaIndex);
		}
		return res;
	}
}
