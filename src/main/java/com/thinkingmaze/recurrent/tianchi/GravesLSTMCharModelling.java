package com.thinkingmaze.recurrent.tianchi;


import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**GravesLSTM Character modelling example
 * @author Alex Black

   Example: Train a LSTM RNN to generates text, one character at a time.
	This example is somewhat inspired by Andrej Karpathy's blog post,
	"The Unreasonable Effectiveness of Recurrent Neural Networks"
	http://karpathy.github.io/2015/05/21/rnn-effectiveness/
	
	Note that this example has not been well tuned - better performance is likely possible with better hyperparameters
	
	Some differences between this example and Karpathy's work:
	- The LSTM architectures appear to differ somewhat. GravesLSTM has peephole connections that
	  Karpathy's char-rnn implementation appears to lack. See GravesLSTM javadoc for details.
	  There are pros and cons to both architectures (addition of peephole connections is a more powerful
	  model but has more parameters per unit), though they are not radically different in practice.
	- Karpathy uses truncated backpropagation through time (BPTT) on full character
	  sequences, whereas this example uses standard (non-truncated) BPTT on partial/subset sequences.
	  Truncated BPTT is probably the preferred method of training for this sort of problem, and is configurable
      using the .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength().tBPTTBackwardLength() options
	  
	This example is set up to train on the Complete Works of William Shakespeare, downloaded
	 from Project Gutenberg. Training on other text sources should be relatively easy to implement.
 */
public class GravesLSTMCharModelling {
	public static void main( String[] args ) throws Exception {
		int lstmLayerSize = 150;					//Number of units in each GravesLSTM layer
		int miniBatchSize = 32;						//Size of mini batch to use when  training
		int examplesPerEpoch = 50 * miniBatchSize;	//i.e., how many examples to learn on between generating samples
		int exampleLength = 70;						//Length of each training example
		int numEpochs = 60;							//Total number of training + sample generation epochs
		// Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
		// Initialization characters must all be in CharacterIterator.getMinimalCharacterSet() by default
		Random rng = new Random(12345);
		
		//Get a DataSetIterator that handles vectorization of text into something we can use to train
		// our GravesLSTM network.
		String trainFilePath = "D:/MyEclipse/alibaba/mars_tianchi_train_data.csv";
		String testFilePath = "D:/MyEclipse/alibaba/mars_tianchi_test_data.csv";
		ArtistIterator iterTrain = new ArtistIterator(trainFilePath, miniBatchSize, examplesPerEpoch, exampleLength, true);
		ArtistIterator iterTest = new ArtistIterator(testFilePath, 1, 50, 70, false);
		
		//Set up network configuration:
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
			.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
			.learningRate(0.03)
			.rmsDecay(0.95)
			.seed(12345)
			.regularization(false)
			.l2(0.001)
			.list(3)
			.layer(0, new GravesLSTM.Builder().nIn(iterTrain.inputColumns()).nOut(lstmLayerSize)
					.updater(Updater.RMSPROP)
					.activation("tanh").weightInit(WeightInit.DISTRIBUTION)
					.dist(new UniformDistribution(-0.08, 0.08)).build())
			.layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
					.updater(Updater.RMSPROP)
					.activation("tanh").weightInit(WeightInit.DISTRIBUTION)
					.dist(new UniformDistribution(-0.08, 0.08)).build())
			.layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation("softmax")        //MCXENT + softmax for classification
					.updater(Updater.RMSPROP)
					.nIn(lstmLayerSize).nOut(iterTrain.totalOutcomes()).weightInit(WeightInit.DISTRIBUTION)
					.dist(new UniformDistribution(-0.08, 0.08)).build())
			.pretrain(false).backprop(true)
			.build();
		
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		//net.setListeners(new ScoreIterationListener(1));
		
		//Print the  number of parameters in the network (and for each layer)
		Layer[] layers = net.getLayers();
		int totalNumParams = 0;
		for( int i=0; i<layers.length; i++ ){
			int nParams = layers[i].numParams();
			System.out.println("Number of parameters in layer " + i + ": " + nParams);
			totalNumParams += nParams;
		}
		System.out.println("Total number of network parameters: " + totalNumParams);
		
		//Do training, and then generate and print samples from network
		for( int i=0; i<numEpochs; i++ ){
			net.fit(iterTrain);
			String predictFilePath = "D:/MyEclipse/alibaba/mars_tianchi_predict_data.csv";
			sampleCharactersFromNetwork(predictFilePath, net, iterTest, 30, rng);
			System.out.println("Complete " + (i+1) + " epoch.");
			System.out.println(Evaluate.f1Value(predictFilePath));
			iterTest.reset();
		}
		
		System.out.println("\n\nExample complete");
	}

	private static void sampleCharactersFromNetwork( String predictFilePath, MultiLayerNetwork net,
			ArtistIterator iter, int featureDays, Random rng ) throws IOException{
		//Set up initialization. If no initialization: use a random character
		FileWriter predictFile = new FileWriter(new File(predictFilePath));
		while(iter.hasNext()){
			//Create input for initialization
			DataSet data = iter.next();
			INDArray input = data.getFeatures();
			int[][] output = iter.getRealOutput();
			for(int k = 0; k<output.length; k++){
				for(int i = 10; i<output[k].length; i++){
					predictFile.write(String.valueOf(output[k][i]));
					if(i+1 == output[k].length)
						predictFile.write("\n");
					else predictFile.write(",");
				}
				INDArray initializationInput = Nd4j.zeros(1, iter.inputColumns(), 11);
				int preChar = -1;
				for(int i = 0; i <= 10; i++){
					for(int j = 0; j < iter.inputColumns(); j++){
						initializationInput.putScalar(new int[]{0,j,i}, input.getDouble(k,j,i));
						if(input.getDouble(k,j,i) > 0) preChar = iter.getCharacterValue(j);
					}
				}
				net.rnnClearPreviousState();
				INDArray out = net.rnnTimeStep(initializationInput);
				out = out.tensorAlongDimension(out.size(2)-1,1,0);	//Gets the last time step output
				for( int i=10; i<output[k].length; i++ ){
					INDArray nextInput = Nd4j.zeros(1,iter.inputColumns());
					double[] outputProbDistribution = new double[iter.totalOutcomes()];
					for( int j=0; j<outputProbDistribution.length; j++ ) {
						outputProbDistribution[j] = out.getDouble(0,j);
					}
					int sampledCharacter = sampleFromDistribution(outputProbDistribution,iter,rng);
					preChar = sampledCharacter;
					predictFile.write(String.valueOf(sampledCharacter));
					if(i+1 == output[k].length)
						predictFile.write("\n");
					else predictFile.write(",");
					nextInput.putScalar(new int[]{0,iter.characterToIndex(sampledCharacter)}, 1.0f);
					out = net.rnnTimeStep(nextInput);
				}
			}
		}
		predictFile.close();
		return ;
	}
	
	/** Given a probability distribution over discrete classes, sample from the distribution
	 * and return the generated class index.
	 * @param distribution Probability distribution over classes. Must sum to 1.0
	 */
	private static int sampleFromDistribution( double[] distribution, ArtistIterator iter, Random rng ){
//		double d = 0;
//		double sum = 0.0;
//		for( int i=0; i<distribution.length; i++ ){
//			sum += distribution[i];
//			d += distribution[i]/iter.getCharacterValue(i);
//		}
//		//Should never happen if distribution is a valid probability distribution
//		if(sum> 1.01 || sum < 0.99)
//			throw new IllegalArgumentException("Distribution is invalid? d="+d+", sum="+sum);
//		return (int) (1.0/d+0.5);
		double d = rng.nextDouble();
		double sum = 0.0;
		for( int i=0; i<distribution.length; i++ ){
			sum += distribution[i];
			if( d <= sum ) return iter.getCharacterValue(i);
		}
		//Should never happen if distribution is a valid probability distribution
		throw new IllegalArgumentException("Distribution is invalid? d="+d+", sum="+sum);
//		int res = 0;
//		for( int i=1; i<distribution.length; i++ ){
//			if(distribution[res] < distribution[i]) 
//				res = i;
//		}
//		return iter.getCharacterValue(res);
	}
}