import activation.Relu;
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
		// BS-IPSO epochs = 5, IPSO-BP = 5
		// BP use https://www.sciencedirect.com/science/article/pii/S0096300306008277
		int epochs = 5;
		int hiddenLayerNeurons = 20;
		double runAvgTime = 0;
		double runAvgMse = 0;

		int dataSize = 500;
		int inputShape = 2;
		int outputShape = 1;
		double[][] feature = new double[dataSize][inputShape];
		double[][] label = new double[feature.length][outputShape];

		// ackley
		double ackleySum1;
		double ackleySum2;

		long timeStart, timeEnd;

		java.util.Random generator = new java.util.Random(0);

		for (int i = 0; i < dataSize; i++)
		{
			// ackley
			ackleySum1 = 0;
			ackleySum2 = 0;
			for (int j = 0; j < inputShape; j++)
			{
				// sin(2x)e^-x problem
				// feature[i][j] = generator.nextDouble() * Math.PI;

				// xor problem
				// feature[i][j] = generator.nextInt(2); // 0~1

				// ackley
				ackleySum1 = ackleySum1 + Math.pow(feature[i][j], 2);
				ackleySum2 = ackleySum2 + Math.cos(2 * Math.PI * feature[i][j]);
			}

			// y = sin(2x)e^-x, inputShape = 1
			// label[i][0] = Math.sin(2 * feature[i][0]) * Math.exp(-1 * feature[i][0]);

			// ackley function, inputShape = 2
			label[i][0] = -20 * Math.exp(-0.2 * Math.sqrt((1 / inputShape) * ackleySum1))
					- Math.exp((1 / inputShape) * ackleySum2) + 20 + Math.exp(1);

			// xor problem, inputShape = 3
			// label[i][0] = 0;
			// for (int j = 0; j < inputShape; j++)
			// {
			// label[i][0] = Math.abs(label[i][0] - feature[i][j]);
			// }
		}

		double[] temp;
		int randomNumber;
		for (int i = 0; i < label.length; i++)
		{
			// randomNumber = (int) (Math.random() * label.length);
			randomNumber = (int) (generator.nextDouble() * label.length);
			temp = feature[i];
			feature[i] = feature[randomNumber];
			feature[randomNumber] = temp;
			temp = label[i];
			label[i] = label[randomNumber];
			label[randomNumber] = temp;
		}

		double[][] trainFeature = new double[(int) (dataSize * 0.7)][inputShape];
		double[][] trainLabel = new double[trainFeature.length][outputShape];
		double[][] testFeature = new double[dataSize - trainFeature.length][inputShape];
		double[][] testLabel = new double[testFeature.length][outputShape];
		double[][] predictLabel = new double[testFeature.length][outputShape];

		for (int i = 0; i < label.length; i++)
		{
			if (i < (int) (dataSize * 0.7))
			{
				trainFeature[i] = feature[i];
				trainLabel[i] = label[i];
			}
			else
			{
				testFeature[i - (int) (dataSize * 0.7)] = feature[i];
				testLabel[i - (int) (dataSize * 0.7)] = label[i];
			}
		}

		AbstractLossFunction loss = new MeanSquaredError();

		for (int i = 0; i < run; i++)
		{
			Model model = new Model();
			double lossValue = 0;
			double avgLossValue = 0;

			model.add(new Layer(hiddenLayerNeurons, new Random(50, -50), new Sigmoid()));
			model.add(new Layer(1, new Random(50, -50), new Relu()));

			// model.compile(inputShape, loss, new BackPropagation(0.01, 10));

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
			// new HybridParticleSwarmOptimizationBackPropagation(psoParameter,
			// trainFeature.length,
			// HybridParticleSwarmOptimizationBackPropagation.FIRST_CONDITION, 1000, 0.01,
			// 0.05));

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
