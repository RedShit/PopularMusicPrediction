package com.thinkingmaze.hmm.train;

import java.util.Arrays;

import com.thinkingmaze.hmm.HMM;
import com.thinkingmaze.hmm.observation.Observation;
import com.thinkingmaze.hmm.version.Version;

public class TrainGLRHMM extends Training {

	private double[][] tAlpha;
	private double[][] tBeta;
	private Version alphaVersion;
	private Version betaVersion;
	
	public TrainGLRHMM(Observation observation, HMM hmm) {
		super.observation = observation;
		super.hmm = hmm;
		super.stateLength = hmm.getStateLength();
		super.deltaLength = hmm.getDeltaLength();
		this.alphaVersion = new Version(-1);
		this.betaVersion = new Version(-1);
	}

	@Override
	public void BaumWelch() {
		super.newTransitMatrix = new double[stateLength][stateLength];
		super.newEmissionMatrix = new double[stateLength][deltaLength];
		super.newPiMatrix = new double[stateLength];
		
		System.out.println("Updating Transit Prob ... ...");
		for (int i = 1; i <= stateLength; i++) {
			for (int j = 1; j <= stateLength; j++) {
				updateTransitProb(i, j);
			}
		}
		
		System.out.println("Updating Emission Prob ... ...");
		for (int i = 1; i <= stateLength; i++) {
			for (int j = 1; j <= deltaLength; j++) {
				updateEmissionProb(i, j);
			}
		}
		
		System.out.println("Updating Pi ... ...");
		for (int i = 1; i <= stateLength; i++) {
			updatePi(i);
		}
		
		finalize();
	}

	public void trainPiMatrix(){
		super.newPiMatrix = new double[stateLength];
		for (int i = 1; i <= stateLength; i++) {
			updatePi(i);
		}
		hmm.setPiMatrix(newPiMatrix);
	}
	
	private double getAlpha(int obsIndex, int stateIndex) {
		if(obsIndex > observation.getVectorLength())
			throw new RuntimeException("obsIndex > observations.getChild(obsNumber).getSize().");
		
		if(this.tAlpha[obsIndex-1][stateIndex-1]>= 0.0)
			return this.tAlpha[obsIndex-1][stateIndex-1];
		if (obsIndex == 1) {
			return this.tAlpha[obsIndex-1][stateIndex-1]=hmm.getPiMatrix(stateIndex)
					* hmm.getEmissionProbMatrix(stateIndex,
							observation.getVector(1));
		}
		double sum = 0;

		for (int j = 1; j <= stateLength; j++) {
			sum += hmm.getStateTransitProbMatrix(j, stateIndex)
					* getAlpha(obsIndex - 1, j);
		}
		return this.tAlpha[obsIndex-1][stateIndex-1]=sum
				* hmm.getEmissionProbMatrix(stateIndex,
						observation.getVector(obsIndex));
		
	}
	
	@Override
	public double alpha(int obsIndex, int stateIndex) {
		if(obsIndex > observation.getVectorLength())
			throw new RuntimeException("obsIndex > observation.getVectorLength().");
		
		if(alphaVersion.getVersion() != hmm.getVersionHMM().getVersion()){
			this.tAlpha = new double[observation.getVectorLength()][];
			for(int i = 0; i < observation.getVectorLength(); i++){
				this.tAlpha[i] = new double[stateLength];
				Arrays.fill(this.tAlpha[i], -1.0);
			}
			alphaVersion.setVersion(hmm.getVersionHMM().getVersion());
		}
		return getAlpha(obsIndex, stateIndex);
	}
	
	private double getBeta(int obsIndex, int stateIndex) {
		
		if(obsIndex > observation.getVectorLength())
			throw new RuntimeException("obsIndex > observation.getVectorLength().");
		if(this.tBeta[obsIndex-1][stateIndex-1]>=0)
			return this.tBeta[obsIndex-1][stateIndex-1];
		if (obsIndex == observation.getVectorLength()) {
			return this.tBeta[obsIndex-1][stateIndex-1]=1;
		}
		
		double sum = 0;
		for (int j = 1; j <= stateLength; j++) {
			sum += hmm.getStateTransitProbMatrix(stateIndex, j)
					* hmm.getEmissionProbMatrix(j, observation.getVector(obsIndex+1))
					* getBeta(obsIndex+1, j);
		}

		return this.tBeta[obsIndex-1][stateIndex-1]=sum;
	}

	@Override
	public double beta(int obsIndex, int stateIndex) {
		if(obsIndex > observation.getVectorLength())
			throw new RuntimeException("obsIndex > observation.getVectorLength().");
		
		if(betaVersion.getVersion() != hmm.getVersionHMM().getVersion()){
			this.tBeta = new double[observation.getVectorLength()][];
			for(int i = 0; i < observation.getVectorLength(); i++){
				this.tBeta[i] = new double[stateLength];
				Arrays.fill(this.tBeta[i], -1.0);
			}
			betaVersion.setVersion(hmm.getVersionHMM().getVersion());
		}
		return getBeta(obsIndex, stateIndex);
	}
	
	public double sigma(int obsIndex, int stateIndex) {
		
		if(obsIndex > observation.getVectorLength())
			throw new RuntimeException("obsIndex > observation.getVectorLength().");
		
		double numerator = alpha(obsIndex, stateIndex)
				* beta(obsIndex, stateIndex);
		double denominator = 0;
		if(numerator <= 0.0)
			return numerator;
		for (int j = 1; j <= stateLength; j++) {
			denominator += alpha(obsIndex, j) * beta(obsIndex, j);
		}
		return numerator / denominator;
	}

	public double kappa(int obsIndex, int stateIndex1, int stateIndex2) {

		double numerator = alpha(obsIndex, stateIndex1)
				* hmm.getStateTransitProbMatrix(stateIndex1, stateIndex2)
				* hmm.getEmissionProbMatrix(stateIndex2, observation.getVector(obsIndex + 1))
				* beta(obsIndex + 1, stateIndex2);

		double denominator = 0;
		
		if(numerator <= 0)
			return 0;
		
		for (int k = 1; k <= stateLength; k++) {
			for (int l = 1; l <= stateLength; l++) {
				denominator += alpha(obsIndex, k)
						* hmm.getStateTransitProbMatrix(k, l)
						* hmm.getEmissionProbMatrix(l, observation.getVector(obsIndex + 1))
						* beta(obsIndex + 1, l);
			}
		}

		return numerator / denominator;
	}

	public void updateTransitProb(int stateIndex1, int stateIndex2) {
		double numerator = 0;
		double denominator = 0;
		double newTransitProb;
		
		for (int t = 1; t <= observation.getVectorLength() - 1; t++) {
			numerator += kappa(t, stateIndex1, stateIndex2);
		}

		for (int t = 1; t <= observation.getVectorLength() - 1; t++) {
			denominator += sigma(t , stateIndex1);
		}
		
		if(denominator <= 0.0)
			throw new RuntimeException("denominator <= 0.0");
		
		newTransitProb = numerator / denominator;

		newTransitMatrix[stateIndex1 - 1][stateIndex2 - 1] = newTransitProb;
	}

	public void updatePi(int stateIndex) {
		newPiMatrix[stateIndex - 1] = sigma(1, stateIndex);
	}

	public void scaleTransitMatrix() {
		super.transitScaleFactor = new double[stateLength];
		for (int i = 0; i < stateLength; i++) {
			for (int j = 0; j < stateLength; j++) {
				transitScaleFactor[i] += newTransitMatrix[i][j];
			}
		}

		for (int i = 0; i < stateLength; i++) {
			if(transitScaleFactor[i] <= 0.0)
				throw new RuntimeException("transitScaleFactor["+i+"] <= 0.0");
			transitScaleFactor[i] = 1 / transitScaleFactor[i];
		}

		for (int i = 0; i < stateLength; i++) {
			for (int j = 0; j < stateLength; j++) {
				newTransitMatrix[i][j] = newTransitMatrix[i][j]
						* transitScaleFactor[i];
			}
		}
	}

	public void scalePiMatrix() {
		piScaleFactor = 0;
		for (int i = 0; i < stateLength; i++) {
			piScaleFactor += newPiMatrix[i];
		}
		
		if(piScaleFactor <= 0.0)
			throw new RuntimeException("piScaleFactor <= 0.0");
		
		piScaleFactor = 1 / piScaleFactor;

		for (int i = 0; i < stateLength; i++) {
			newPiMatrix[i] = newPiMatrix[i] * piScaleFactor;
		}
	}

	public void printUpdate() {
		System.out.println("[State Transition Probability Matrix]");

		for (int i = 0; i < stateLength; i++) {
			for (int j = 0; j < stateLength; j++) {
				System.out.printf("%-7.3f",
						hmm.getStateTransitProbMatrix(i + 1, j + 1));
			}
			System.out.println();
		}
		
		System.out.println();
		System.out.println("[Emission Probability Matrix]");

		for (int i = 0; i < stateLength; i++) {
			for (int j = 0; j < deltaLength; j++) {
				System.out.printf("%-7.3f",
						hmm.getEmissionProbMatrix(i + 1, j + 1));
			}
			System.out.println();
		}
		
		System.out.println();
		System.out.println("[Pi Matrix]");
		for (int j = 0; j < stateLength; j++) {
			System.out.printf("%-7.3f", hmm.getPiMatrix(j + 1));
		System.out.println();
		}
	}

	public void finalize() {
		hmm.setStateTransitProbMatrix(newTransitMatrix);
		hmm.setEmissionProbMatrix(newEmissionMatrix);
		hmm.setPiMatrix(newPiMatrix);
		//printUpdate();
	}
	
	public double forward() {
		// TODO Auto-generated method stub
		double sum = 0.0;	
		for(int s = 1; s <= this.stateLength; s++){
			sum += alpha(observation.getVectorLength(),s);
		}
		return sum;
	}

	@Override
	public void updateEmissionProb(int stateIndex, int vectorIndex) {
		// TODO Auto-generated method stub
		double numerator = 0;
		double denominator = 0;
		double newEmissionProb;

		for (int t = 1; t <= observation.getVectorLength(); t++) {
			if (observation.getVector(t) == vectorIndex) {
				numerator += sigma(t, stateIndex);
			}
		}

		for (int t = 1; t <= observation.getVectorLength(); t++) {
			denominator += sigma(t, stateIndex);
		}

		if(denominator <= 0.0)
			throw new RuntimeException("denominator <= 0.0");
		
		newEmissionProb = numerator / denominator;

		newEmissionMatrix[stateIndex - 1][vectorIndex - 1] = newEmissionProb;
	}

	@Override
	public void scaleEmissionMatrix() {
		// TODO Auto-generated method stub
		super.emissionScaleFactor = new double[stateLength];
		for (int i = 0; i < stateLength; i++) {
			for (int j = 0; j < deltaLength; j++) {
				emissionScaleFactor[i] += newEmissionMatrix[i][j];
			}
		}

		for (int i = 0; i < stateLength; i++) {
			if(emissionScaleFactor[i] <= 0.0)
				throw new RuntimeException("emissionScaleFactor["+i+"] <= 0.0");
			emissionScaleFactor[i] = 1 / emissionScaleFactor[i];
		}

		for (int i = 0; i < stateLength; i++) {
			for (int j = 0; j < deltaLength; j++) {
				newEmissionMatrix[i][j] = newEmissionMatrix[i][j]
						* emissionScaleFactor[i];
			}
		}
	}
}
