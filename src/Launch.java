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
		int dataSize = 100;
		int inputShape = 1;
		int outputShape = 1;
		double[][] feature = new double[dataSize][inputShape];
		double[][] label = new double[feature.length][outputShape];

		double ackleySum1;
		double ackleySum2;

		java.util.Random generator = new java.util.Random(0);

		for (int i = 0; i < dataSize; i++)
		{
			ackleySum1 = 0;
			ackleySum2 = 0;
			for (int j = 0; j < inputShape; j++)
			{
				// feature[i][j] = Math.random() * 30 * 2 - 30;
				feature[i][j] = generator.nextDouble() * 30 * 2 - 30;
				ackleySum1 = ackleySum1 + Math.pow(feature[i][j], 2);
				ackleySum2 = ackleySum2 + Math.cos(2 * Math.PI * feature[i][j]);
			}
			// y = 2*x^2 + 1
			// label[i][0] = 2 * Math.pow(feature[i][0], 2) + 1;

			// y = 2*x + 1
			// label[i][0] = 2 * feature[i][0] + 1;

			// ackley function
			label[i][0] = -20 * Math.exp(-0.2 * Math.sqrt((1 / inputShape) * ackleySum1))
					- Math.exp((1 / inputShape) * ackleySum2) + 20 + Math.exp(1);
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

		Model model = new Model();
		model.add(new Layer(5, new Random(), new Sigmoid()));
		model.add(new Layer(1, new Random(), new Relu()));
		// model.compile(inputShape, loss, new BackPropagation(0.01, 10));
		model.compile(inputShape, loss,
				new BatchParticleSwarmOptimization(100, 0.8, 0.5, 1.2, 0.5, 99999, 10 * dataSize / 4, 1, 10));
		model.fit(trainFeature, trainLabel, 10);
		for (int i = 0; i < testFeature.length; i++)
		{
			predictLabel[i] = model.predict(testFeature[i]);
		}
		for (int i = 0; i < predictLabel.length; i++)
		{
			System.out.println(i + "th final mse:" + loss.getError(predictLabel[i], testLabel[i]));
		}
	}
}
