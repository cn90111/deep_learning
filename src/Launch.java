import java.io.FileNotFoundException;

import activation.Relu;
import activation.Softmax;
import dataset.Dataset;
import dataset.Mnist;
import initializer.Random;
import layer.Layer;
import loss.AbstractLossFunction;
import loss.CrossEntropy;
import model.Model;
import optimizer.BackPropagation;

public class Launch
{
	public static void main(String[] args) throws FileNotFoundException
	{
		int run = 30;
		int epochs = 20;
		int hiddenLayerNeurons = 512;
		int batchSize = 128;
		double learningRate = 0.01;

		Dataset dataset = new Mnist();

		double[][] trainFeature = dataset.getTrainFeature();
		double[][] trainLabel = dataset.getTrainLabel();
		double[][] testFeature = dataset.getTestFeature();
		double[][] testLabel = dataset.getTestLabel();
		double[][] predictLabel = new double[testFeature.length][testFeature[0].length];

		double runAvgTime = 0;
		double runAvgMse = 0;

		double lossValue = 0;
		double avgLossValue = 0;

		long timeStart, timeEnd;

		AbstractLossFunction loss = new CrossEntropy();

		for (int i = 0; i < run; i++)
		{
			Model model = new Model();
			model.add(new Layer(hiddenLayerNeurons, new Random(1, 0), new Relu()));
			model.add(new Layer(hiddenLayerNeurons, new Random(1, 0), new Relu()));
			model.add(new Layer(trainLabel[0].length, new Random(1, 0), new Softmax()));

			model.compile(trainFeature[0].length, loss, new BackPropagation(learningRate, batchSize));

			timeStart = System.currentTimeMillis();
			model.fit(trainFeature, trainLabel, epochs, true);
			timeEnd = System.currentTimeMillis();

			for (int j = 0; j < testFeature.length; j++)
			{
				predictLabel[j] = model.predict(testFeature[j]);
			}
			for (int j = 0; j < predictLabel.length; j++)
			{
				lossValue = loss.getError(predictLabel[j], testLabel[j]);
				avgLossValue = avgLossValue + lossValue;
			}

			avgLossValue = avgLossValue / predictLabel.length;
			runAvgMse = runAvgMse + avgLossValue;

			System.out.println(i + " run loss value : " + avgLossValue);
			System.out.println(i + " run train time : " + (timeEnd - timeStart) + " ms");
			runAvgTime = runAvgTime + (timeEnd - timeStart);
		}
		runAvgMse = runAvgMse / run;
		runAvgTime = runAvgTime / run;

		System.out.println("avg " + run + " run mse : " + runAvgMse);
		System.out.println("avg " + run + " run train time : " + runAvgTime + " ms");
	}
}
