package com.thinkingmaze.hmm.train;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.thinkingmaze.hmm.HMM;
import com.thinkingmaze.hmm.observation.Observation;
import com.thinkingmaze.hmm.observation.Observations;
import com.thinkingmaze.hmm.version.Version;

public class TrainGLRHMM extends Training {

	private double[][][] tAlpha;
	private double[][][] tBeta;
	private List<Version> alphaVersion;
	private List<Version> betaVersion;
	
	public TrainGLRHMM(Observations observations, HMM hmm) {
		
		super.observations = observations;
		super.hmm = hmm;
		super.stateLength = hmm.getStateLength();
		super.deltaLength = hmm.getDeltaLength();
		
		this.alphaVersion = new ArrayList<Version>();
		for(int i = 0; i < observations.getSize(); i++){
			this.alphaVersion.add(new Version(-1));
		}
		this.betaVersion = new ArrayList<Version>();
		for(int i = 0; i < observations.getSize(); i++){
			this.betaVersion.add(new Version(-1));
		}
		this.tAlpha = new double[observations.getSize()][][];
		this.tBeta = new double[observations.getSize()][][];
	}

	@Override
	public void BaumWelch() {
		super.newTransitMatrix = new double[stateLength][stateLength];
		super.newEmissionMatrix = new double[stateLength][deltaLength];
		super.newPiMatrix = new double[observations.getSize()][stateLength];
		
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

	private double getAlpha(int obsNumber, int obsIndex, int stateIndex) {
		
		if(obsNumber > observations.getSize())
			throw new RuntimeException("obsNumber > observations.getSize().");
		if(obsIndex > observations.getChild(obsNumber).getVectorLength())
			throw new RuntimeException("obsIndex > observations.getChild(obsNumber).getSize().");
		if(this.tAlpha[obsNumber-1][obsIndex-1][stateIndex-1]>= 0.0)
			return this.tAlpha[obsNumber-1][obsIndex-1][stateIndex-1];
		if (obsIndex == 1) {
			return this.tAlpha[obsNumber-1][obsIndex-1][stateIndex-1]=hmm.getPiMatrix(obsNumber, stateIndex)
					* hmm.getEmissionProbMatrix(stateIndex,
							observations.getChild(obsNumber).getVector(1));
		}
		double sum = 0;

		for (int j = 1; j <= stateLength; j++) {
			sum += hmm.getStateTransitProbMatrix(j, stateIndex)
					* getAlpha(obsNumber, obsIndex - 1, j);
		}
		return this.tAlpha[obsNumber-1][obsIndex-1][stateIndex-1]=sum
				* hmm.getEmissionProbMatrix(stateIndex,
						observations.getChild(obsNumber).getVector(obsIndex));
		
	}
	
	@Override
	public double alpha(int obsNumber, int obsIndex, int stateIndex) {
		if(obsNumber > observations.getSize())
			throw new RuntimeException("obsNumber > observations.getSize().");
		if(obsIndex > observations.getChild(obsNumber).getVectorLength())
			throw new RuntimeException("obsIndex > observations.getChild(obsNumber).getSize().");
		
		if(alphaVersion.get(obsNumber-1).getVersion() != hmm.getVersionHMM().getVersion()){
			this.tAlpha[obsNumber-1] = new double[observations.getChild(obsNumber).getVectorLength()][];
			for(int i = 0; i < observations.getChild(obsNumber).getVectorLength(); i++){
				this.tAlpha[obsNumber-1][i] = new double[stateLength];
				Arrays.fill(this.tAlpha[obsNumber-1][i], -1.0);
			}
			alphaVersion.get(obsNumber-1).setVersion(hmm.getVersionHMM().getVersion());
		}
		return getAlpha(obsNumber, obsIndex, stateIndex);
	}
	
	private double getBeta(int obsNumber, int obsIndex, int stateIndex) {
		
		if(obsNumber > observations.getSize())
			throw new RuntimeException("obsNumber > observations.getSize().");
		if(obsIndex > observations.getChild(obsNumber).getVectorLength())
			throw new RuntimeException("obsIndex > observations.getChild(obsNumber).getSize().");
		if(this.tBeta[obsNumber-1][obsIndex-1][stateIndex-1]>=0)
			return this.tBeta[obsNumber-1][obsIndex-1][stateIndex-1];
		if (obsIndex == observations.getChild(obsNumber).getVectorLength()) {
			return this.tBeta[obsNumber-1][obsIndex-1][stateIndex-1]=1;
		}
		
		double sum = 0;
		for (int j = 1; j <= stateLength; j++) {
			sum += hmm.getStateTransitProbMatrix(stateIndex, j)
					* hmm.getEmissionProbMatrix(j,
							observations.getChild(obsNumber).getVector(obsIndex+1))
					* getBeta(obsNumber, obsIndex+1, j);
		}

		return this.tBeta[obsNumber-1][obsIndex-1][stateIndex-1]=sum;
	}

	@Override
	public double beta(int obsNumber, int obsIndex, int stateIndex) {
		
		if(obsNumber > observations.getSize())
			throw new RuntimeException("obsNumber > observations.getSize().");
		if(obsIndex > observations.getChild(obsNumber).getVectorLength())
			throw new RuntimeException("obsIndex > observations.getChild(obsNumber).getSize().");
		
		if(betaVersion.get(obsNumber-1).getVersion() != hmm.getVersionHMM().getVersion()){
			this.tBeta[obsNumber-1] = new double[observations.getChild(obsNumber).getVectorLength()][];
			for(int i = 0; i < observations.getChild(obsNumber).getVectorLength(); i++){
				this.tBeta[obsNumber-1][i] = new double[stateLength];
				Arrays.fill(this.tBeta[obsNumber-1][i], -1.0);
			}
			betaVersion.get(obsNumber-1).setVersion(hmm.getVersionHMM().getVersion());
		}
		return getBeta(obsNumber, obsIndex, stateIndex);
	}
	
	public double sigma(int obsNumber, int obsIndex, int stateIndex) {
		
		if(obsNumber > observations.getSize())
			throw new RuntimeException("obsNumber > observations.getSize().");
		if(obsIndex > observations.getChild(obsNumber).getVectorLength())
			throw new RuntimeException("obsIndex > observations.getChild(obsNumber).getSize().");
		
		double numerator = alpha(obsNumber, obsIndex, stateIndex)
				* beta(obsNumber, obsIndex, stateIndex);
		double denominator = 0;
		if(numerator <= 0.0)
			return numerator;
		for (int j = 1; j <= stateLength; j++) {
			denominator += alpha(obsNumber, obsIndex, j) * beta(obsNumber, obsIndex, j);
		}
		return numerator / denominator;
	}

	public double kappa(int obsNumber, int obsIndex, int stateIndex1, int stateIndex2) {

		double numerator = alpha(obsNumber, obsIndex, stateIndex1)
				* hmm.getStateTransitProbMatrix(stateIndex1, stateIndex2)
				* hmm.getEmissionProbMatrix(stateIndex2, observations.getChild(obsNumber).getVector(obsIndex + 1))
				* beta(obsNumber, obsIndex + 1, stateIndex2);

		double denominator = 0;
		
		if(numerator <= 0)
			return 0;
		
		for (int k = 1; k <= stateLength; k++) {
			for (int l = 1; l <= stateLength; l++) {
				denominator += alpha(obsNumber, obsIndex, k)
						* hmm.getStateTransitProbMatrix(k, l)
						* hmm.getEmissionProbMatrix(l, observations.getChild(obsNumber).getVector(obsIndex + 1))
						* beta(obsNumber, obsIndex + 1, l);
			}
		}

		return numerator / denominator;
	}

	public void updateTransitProb(int stateIndex1, int stateIndex2) {
		double numerator = 0;
		double denominator = 0;
		double newTransitProb;
		
		for (int n = 1; n <= observations.getSize(); n++){
			for (int t = 1; t <= observations.getChild(n).getVectorLength() - 1; t++) {
				numerator += kappa(n, t, stateIndex1, stateIndex2);
			}
		}
		for (int n = 1; n <= observations.getSize(); n++){
			for (int t = 1; t <= observations.getChild(n).getVectorLength() - 1; t++) {
				Double x = sigma(n, t , stateIndex1);
				if(x.isNaN())
					throw new RuntimeException("sigma(n, t , stateIndex1) is NaN.");
				denominator += sigma(n, t , stateIndex1);
			}
		}
		
		if(denominator <= 0.0)
			throw new RuntimeException("denominator <= 0.0");
		
		newTransitProb = numerator / denominator;

		newTransitMatrix[stateIndex1 - 1][stateIndex2 - 1] = newTransitProb;
	}

	public void updateEmissionProb(int stateIndex, int vectorIndex) {
		double numerator = 0;
		double denominator = 0;
		double newEmissionProb;

		for (int n = 1; n <= observations.getSize(); n++){
			for (int t = 1; t <= observations.getChild(n).getVectorLength(); t++) {
				if (observations.getChild(n).getVector(t) == vectorIndex) {
					numerator += sigma(n, t, stateIndex);
				}
			}
		}

		for (int n = 1; n <= observations.getSize(); n++){
			for (int t = 1; t <= observations.getChild(n).getVectorLength(); t++) {
				denominator += sigma(n, t, stateIndex);
			}
		}

		if(denominator <= 0.0)
			throw new RuntimeException("denominator <= 0.0");
		
		newEmissionProb = numerator / denominator;

		newEmissionMatrix[stateIndex - 1][vectorIndex - 1] = newEmissionProb;
	}

	public void updatePi(int stateIndex) {
		for (int n = 1; n <= observations.getSize(); n++){
			newPiMatrix[n-1][stateIndex - 1] = sigma(n, 1, stateIndex);
		}
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

	public void scaleEmissionMatrix() {
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

	public void scalePiMatrix() {
		piScaleFactor = 0;
		for(int n = 0; n < observations.getSize(); n++){
			for (int i = 0; i < stateLength; i++) {
				piScaleFactor += newPiMatrix[n][i];
			}
			
			if(piScaleFactor <= 0.0)
				throw new RuntimeException("piScaleFactor <= 0.0");
			
			piScaleFactor = 1 / piScaleFactor;
	
			for (int i = 0; i < stateLength; i++) {
				newPiMatrix[n][i] = newPiMatrix[n][i] * piScaleFactor;
			}
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
		if(observations.getSize() > 10){
			System.out.println("omit ... ...");
		}
		else{
			for (int i = 0; i < observations.getSize(); i++){
				for (int j = 0; j < stateLength; j++) {
					System.out.printf("%-7.3f", hmm.getPiMatrix(i + 1, j + 1));
				}
				System.out.println();
			}
		}
	}

	public void finalize() {
		hmm.setStateTransitProbMatrix(newTransitMatrix);
		hmm.setEmissionProbMatrix(newEmissionMatrix);
		hmm.setPiMatrix(newPiMatrix);
		printUpdate();
	}
	
	public double forward() {
		// TODO Auto-generated method stub
		double min = 1.0;
		for(int i = 1; i <= observations.getSize(); i++){
			Observation  observation = (Observation) observations.getChild(i);
			double sum = 0.0;
			for(int s = 1; s <= this.stateLength; s++){
				sum += alpha(i,observation.getVectorLength(),s);
			}
			min = Math.min(min, sum);
		}
		return min;
	}
}
