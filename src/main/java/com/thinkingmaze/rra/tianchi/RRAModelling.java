package com.thinkingmaze.rra.tianchi;


import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

import com.thinkingmaze.rra.train.TrainLinear;
import com.thinkingmaze.rra.train.Training;
import com.thinkingmaze.rrat.LS;

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
public class RRAModelling {
	public static void main( String[] args ) throws Exception {
		Scanner artistSin = new Scanner(new File("D:/MyEclipse/alibaba/mars_tianchi_artist_id.csv"));
		String filePath = "D:/MyEclipse/alibaba/mars_tianchi_test_data.csv";
		ArtistIterator iter = new ArtistIterator(filePath, "all");
		double f1 = 0, f2 = 0;
		while(artistSin.hasNext()){
			String artistId = artistSin.next();
			LS ls = new LS(0.5);
			Training model = new TrainLinear(ls);
			List<Integer> data = iter.getActionList(artistId);
			List<Integer> trainData = new ArrayList<Integer>();
			List<Integer> testData = new ArrayList<Integer>();
			List<Integer> mid = new ArrayList<Integer>();
			List<Integer> predictData = new ArrayList<Integer>();
			for(int i = 0; i < data.size(); i++){
				System.out.print(data.get(i) + ",");
				if(i < 165) continue;
				if(i>=165 && i<=175) mid.add(data.get(i));
				if(i <= 175) trainData.add(data.get(i));
				else testData.add(data.get(i));
			}
			System.out.print("1\n");
			Collections.sort(mid);
			System.out.println(artistId);
			double e1 = model.error(trainData);
			System.out.println("Before Train Error is " + e1);
			model.train(trainData);
			double e2 = model.error(trainData);
			System.out.println("After Train Error is " + e2);
			for(int i = 0; i < data.size(); i++){
				if(i <= 175) continue;
				if(e1 > e2*1.2)
					predictData.add((int) (model.median(trainData)+model.output(i-170)+0.5));
				else
					predictData.add((int) (mid.get(4)));
			}
			f1 += Evaluate.f1Value(predictData, testData);
			predictData = new ArrayList<Integer>();
			for(int i = 0; i < data.size(); i++){
				if(i <= 175) continue;
				predictData.add(mid.get(5));
			}
			f2 += Evaluate.f1Value(predictData, testData);
			
			System.out.println(ls.toString());
			System.out.println("");
//			break;
		}
		System.out.println("F1 Value is " + f1 + ", F2 Value is " + f2);
		artistSin.close();
		System.out.println("\n\n complete");
	}
}