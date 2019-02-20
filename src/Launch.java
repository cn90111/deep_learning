import activation.Linear;
import activation.Sigmoid;
import initializer.Random;
import layer.Layer;
import loss.AbstractLossFunction;
import loss.MeanSquaredError;
import model.Model;
import optimizer.BatchParticleSwarmOptimization;

public class Launch
{
	public static void main(String[] args)
	{
		int run = 5;
		// epochs setting
		// https://www.sciencedirect.com/science/article/pii/S0096300306008277
		int epochs = 267;
		int hiddenLayerNeurons = 4;
		double runAvgTime = 0;
		double runAvgMse = 0;

		// xor problem datasize = 8
		// y = sin(2x)e^-x datasize = training data + testing data = 105 + 32 = 137
		int dataSize = 137;
		int inputShape = 1;
		int outputShape = 1;

		double[][] feature = new double[dataSize][inputShape];
		double[][] label = new double[feature.length][outputShape];

		// ackley
		// double ackleySum1;
		// double ackleySum2;

		// sin(2x)e^-x problem
		double samplingNumber = 0;
		for (int i = 0; i < 105; i++)
		{
			feature[i][0] = samplingNumber;
			samplingNumber = samplingNumber + 0.03;
		}

		samplingNumber = 0.02;
		for (int i = 105; i < feature.length; i++)
		{
			feature[i][0] = samplingNumber;
			samplingNumber = samplingNumber + 0.1;
		}

		// xor problem
		// double[][] xorDataSet = new double[][]
		// {
		// { 0, 0, 0 },
		// { 0, 0, 1 },
		// { 0, 1, 0 },
		// { 0, 1, 1 },
		// { 1, 0, 0 },
		// { 1, 0, 1 },
		// { 1, 1, 0 },
		// { 1, 1, 1 } };

		long timeStart, timeEnd;

		java.util.Random generator = new java.util.Random(0);

		for (int i = 0; i < dataSize; i++)
		{
			// ackley
			// ackleySum1 = 0;
			// ackleySum2 = 0;
			for (int j = 0; j < inputShape; j++)
			{
				// xor problem
				// feature[i][j] = xorDataSet[i][j];

				// ackley
				// ackleySum1 = ackleySum1 + Math.pow(feature[i][j], 2);
				// ackleySum2 = ackleySum2 + Math.cos(2 * Math.PI * feature[i][j]);
			}

			// y = sin(2x)e^-x, inputShape = 1
			label[i][0] = Math.sin(2 * feature[i][0]) * Math.exp(-1 * feature[i][0]);

			// ackley function, inputShape = 2
			// label[i][0] = -20 * Math.exp(-0.2 * Math.sqrt((1 / inputShape) * ackleySum1))
			// - Math.exp((1 / inputShape) * ackleySum2) + 20 + Math.exp(1);

			// xor problem, inputShape = 3
			// label[i][0] = 0;
			// for (int j = 0; j < inputShape; j++)
			// {
			// label[i][0] = Math.abs(label[i][0] - feature[i][j]);
			// }
		}

		// shuffle
		// y = sin(2x)e^-x close
		// double[] temp;
		// int randomNumber;
		// for (int i = 0; i < label.length; i++)
		// {
		// randomNumber = (int) (generator.nextDouble() * label.length);
		// temp = feature[i];
		// feature[i] = feature[randomNumber];
		// feature[randomNumber] = temp;
		// temp = label[i];
		// label[i] = label[randomNumber];
		// label[randomNumber] = temp;
		// }

		// General problem 7:3
		// double[][] trainFeature = new double[(int) (dataSize * 0.7)][inputShape];
		// double[][] trainLabel = new double[trainFeature.length][outputShape];
		// double[][] testFeature = new double[dataSize -
		// trainFeature.length][inputShape];
		// double[][] testLabel = new double[testFeature.length][outputShape];
		// double[][] predictLabel = new double[testFeature.length][outputShape];

		// y = sin(2x)e^-x
		double[][] trainFeature = new double[105][inputShape];
		double[][] trainLabel = new double[trainFeature.length][outputShape];
		double[][] testFeature = new double[32][inputShape];
		double[][] testLabel = new double[testFeature.length][outputShape];
		double[][] predictLabel = new double[testFeature.length][outputShape];

		// xor problem
		// double[][] trainFeature = new double[dataSize][inputShape];
		// double[][] trainLabel = new double[trainFeature.length][outputShape];
		// double[][] testFeature = new double[dataSize][inputShape];
		// double[][] testLabel = new double[testFeature.length][outputShape];
		// double[][] predictLabel = new double[testFeature.length][outputShape];

		for (int i = 0; i < label.length; i++)
		{
			// General problem 7:3
			// if (i < (int) (dataSize * 0.7))
			// {
			// trainFeature[i] = feature[i];
			// trainLabel[i] = label[i];
			// }
			// else
			// {
			// testFeature[i - (int) (dataSize * 0.7)] = feature[i];
			// testLabel[i - (int) (dataSize * 0.7)] = label[i];
			// }

			// y = sin(2x)e^-x
			if (i < 105)
			{
				trainFeature[i] = feature[i];
				trainLabel[i] = label[i];
			}
			else
			{
				testFeature[i - 105] = feature[i];
				testLabel[i - 105] = label[i];
			}

			// xor problem
			// trainFeature[i] = feature[i];
			// trainLabel[i] = label[i];
			// testFeature[i] = feature[i];
			// testLabel[i] = label[i];
		}

		AbstractLossFunction loss = new MeanSquaredError();

		for (int i = 0; i < run; i++)
		{
			Model model = new Model();
			double lossValue = 0;
			double avgLossValue = 0;

			model.add(new Layer(hiddenLayerNeurons, new Random(50, -50), new Sigmoid()));
			model.add(new Layer(1, new Random(50, -50), new Linear()));

			// model.compile(inputShape, loss, new BackPropagation(0.01, 1));

			// velocity : +- velocityLimit / 10
			// solution : 1 ~ -1
			pso.Parameter psoParameter = new pso.Parameter(200, 2.0, 2.0, 1.8, 10, 1, 0, 99999, 1, 0, 10 * dataSize / 4,
					1);
			model.compile(inputShape, loss, new BatchParticleSwarmOptimization(psoParameter, 20));

			// velocity : 1 ~ 0
			// solution : 1 ~ 0
			// IPSO-BP batch size = trainFeature.length
			// BS-IPSO-BP batch size = 20
			// BS-IPSO-BP need open resetBatch()
			// pso.Parameter psoParameter = new pso.Parameter(200, 2.0, 2.0, 1.8, 10, 1, 0,
			// 99999, 1, 0, 10 * dataSize / 4,
			// 1);
			// model.compile(inputShape, loss,
			// new HybridParticleSwarmOptimizationBackPropagation(psoParameter, 20,
			// HybridParticleSwarmOptimizationBackPropagation.FIRST_CONDITION, 2000, 100,
			// 0.01, 0.05));

			timeStart = System.currentTimeMillis();
			model.fit(trainFeature, trainLabel, epochs);
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
