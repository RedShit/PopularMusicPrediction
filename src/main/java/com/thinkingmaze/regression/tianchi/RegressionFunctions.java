package com.thinkingmaze.regression.tianchi;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import com.thinkingmaze.rra.tianchi.Evaluate;

import javax.swing.*;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

/**Example: Train a network to reproduce certain mathematical functions, and plot the results.
 * Plotting of the network output occurs every 'plotFrequency' epochs. Thus, the plot shows the accuracy of the network
 * predictions as training progresses.
 * A number of mathematical functions are implemented here.
 * Note the use of the identity function on the network output layer, for regression
 *
 * @author Alex Black
 */
public class RegressionFunctions {

    public enum Function {Sin, SinXDivX, SquareWave, TriangleWave, Sawtooth};

    //Random number generator seed, for reproducability
    public static final int seed = 12345;
    //Median number
    public static final int median = 2;
    //Number of iterations per minibatch
    public static final int iterations = 1;
    //Number of epochs (full passes of the data)
    public static final int nEpochs = 2400;
    //How frequently should we plot the network output?
    public static final int plotFrequency = 400;
    //Batch size: i.e., each epoch has nSamples/batchSize parameter updates
    public static final int batchSize = 10;
    //Network learning rate
    public static final double learningRate = 0.000001;
    public static final Random rng = new Random(seed);


    public static void main(String[] args) throws IOException{

    	Scanner artistSin = new Scanner(new File("D:/MyEclipse/alibaba/mars_tianchi_artist_id.csv"));
		String filePath = "D:/MyEclipse/alibaba/mars_tianchi_test_data.csv";
		ArtistIterator iter = new ArtistIterator(filePath, "all");
		double f1 = 0;
		while(artistSin.hasNext()){
			String artistId = artistSin.next();
			List<Integer> data = iter.getActionList(artistId);
			List<Integer> trainData = new ArrayList<Integer>();
			List<Integer> testData = new ArrayList<Integer>();
			List<Integer> predictData = new ArrayList<Integer>();
			for(int i = 0; i < data.size(); i++){
				if(i <= 120) trainData.add(data.get(i));
				else testData.add(data.get(i));
			}
			System.out.println(artistId);
	    	
			
	        //Switch these two options to do different functions with different networks
	        boolean useSimpleNetwork = true;   //If true: Network with 1 hidden layer of size 20. False: 2 hidden layers of size 50
	
	        //Generate the training data
	        INDArray trainX = Nd4j.create(trainData.size(),1);
	        INDArray trainY = Nd4j.create(trainData.size(),1);
	        for(int i = 0; i < trainData.size(); i++){
	        	trainX.putScalar(new int[]{i,0}, i);
	        	trainY.putScalar(new int[]{i,0}, curve(trainData, i)/200.0);
	        }
	        DataSetIterator iterator = getTrainingData(trainX,trainY,batchSize,rng);
	
	        //Generate the testing data
	        INDArray testX = Nd4j.create(testData.size(),1);
	        INDArray testY = Nd4j.create(testData.size(),1);
	        for(int i = 0; i < testData.size(); i++){
	        	testX.putScalar(new int[]{i,0}, i+trainData.size());
	        	testY.putScalar(new int[]{i,0}, testData.get(i)/200.0);
	        }
	        
	        //Create the network
	        MultiLayerNetwork net = new MultiLayerNetwork(getNetworkConfiguration(useSimpleNetwork));
	        net.init();
	        net.setListeners(new ScoreIterationListener(1));
	
	
	        //Train the network on the full data set, and evaluate in periodically
	        INDArray[] networkPredictions = new INDArray[nEpochs/ plotFrequency];
	        for( int i=0; i<nEpochs; i++ ){
	            iterator.reset();
	            net.fit(iterator);
	            if((i+1) % plotFrequency == 0) networkPredictions[i/ plotFrequency] = net.output(testX, false);
	        }
	        for( int i = 0; i < testData.size(); i++){
	        	predictData.add((int) (networkPredictions[nEpochs/plotFrequency-1].getDouble(i)*200.0));
	        }
	        f1 += Evaluate.f1Value(predictData, testData);
	        //Plot the target data and the network predictions
	        plot(Evaluate.f1Value(predictData, testData), artistId,testX,testY,networkPredictions);
		}
		System.out.println("F1 Value is " + f1);
		System.out.println("\n\n Completed");
        artistSin.close();
    }

    /**Returns the network configuration
     * @param simple If true: return a simple network (1 hidden layer of size 20). If false: 2 hidden layers of size 50
     */
    public static MultiLayerConfiguration getNetworkConfiguration(boolean simple){
        int numInputs = 1;
        int numOutputs = 1;

        if(simple) {
            int numHiddenNodes = 20;
            return new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .iterations(iterations)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .learningRate(learningRate)
                    .updater(Updater.NESTEROVS).momentum(0.9)
                    .list(2)
                    .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                            .weightInit(WeightInit.XAVIER)
                            .activation("tanh")
                            .build())
                    .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                            .weightInit(WeightInit.XAVIER)
                            .activation("identity").weightInit(WeightInit.XAVIER)
                            .nIn(numHiddenNodes).nOut(numOutputs).build())
                    .pretrain(false).backprop(true).build();
        } else {
            int numHiddenNodes = 50;
            return new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .iterations(iterations)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .learningRate(learningRate)
                    .updater(Updater.NESTEROVS).momentum(0.9)
                    .list(3)
                    .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                            .weightInit(WeightInit.XAVIER)
                            .activation("tanh")
                            .build())
                    .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                            .weightInit(WeightInit.XAVIER)
                            .activation("tanh")
                            .build())
                    .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                            .weightInit(WeightInit.XAVIER)
                            .activation("identity").weightInit(WeightInit.XAVIER)
                            .nIn(numHiddenNodes).nOut(numOutputs).build())
                    .pretrain(false).backprop(true).build();
        }
    }

    /** Create a DataSetIterator for training
     * @param batchSize Batch size (number of examples for every call of DataSetIterator.next())
     * @param rng Random number generator (for repeatability)
     */
    private static DataSetIterator getTrainingData(INDArray x, INDArray y, int batchSize, Random rng){
        DataSet allData = new DataSet(x,y);

        List<DataSet> list = allData.asList();
        Collections.shuffle(list,rng);
        return new ListDataSetIterator(list,batchSize);
    }
    
    private static int curve(List<Integer> data, int x){
    	List<Integer> t = new ArrayList<Integer>();
    	for(int i = Math.max(0, x-median); i < Math.min(x+median, data.size()); i++){
    		t.add(data.get(i));
    	}
    	Collections.sort(t);
    	return t.get(t.size()/2);
    }

    //Plot the data
    private static void plot(double f1, String artistId, INDArray x, INDArray y, INDArray... predicted){
        XYSeriesCollection dataSet = new XYSeriesCollection();
        addSeries(dataSet,x,y,"True Function (Labels)");

        for( int i=0; i<predicted.length; i++ ){
            addSeries(dataSet,x,predicted[i],String.valueOf(i));
        }

        JFreeChart chart = ChartFactory.createXYLineChart(
                "Regression Example - " + artistId + " f1 " + f1,      // chart title
                "X",                      // x axis label
                "Y",     			      // y axis label
                dataSet,                  // data
                PlotOrientation.VERTICAL,
                true,                     // include legend
                true,                     // tooltips
                false                     // urls
        );

        ChartPanel panel = new ChartPanel(chart);

        JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();

        f.setVisible(true);
    }

    private static void addSeries(XYSeriesCollection dataSet, INDArray x, INDArray y, String label){
        double[] xd = x.data().asDouble();
        double[] yd = y.data().asDouble();
        XYSeries s = new XYSeries(label);
        for( int j=0; j<xd.length; j++ ) s.add(xd[j],yd[j]);
        dataSet.addSeries(s);
    }
}
