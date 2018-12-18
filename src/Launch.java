import activation.Relu;
import activation.Sigmoid;
import initializer.Random;
import layer.Layer;
import loss.AbstractLoss;
import loss.MeanSquaredError;
import model.Model;
import optimizer.BackPropagation;

public class Launch
{
	public static void main(String[] args)
	{
		int dataSize = 100;
		int inputShape = 1;
		int outputShape = 1;
		double[][] feature = new double[dataSize][inputShape];
		double[][] label = new double[feature.length][outputShape];

		// y = 2*x^2 + 1
		for (int i = 0; i < feature.length; i++)
		{
			feature[i][0] = Math.random() * 10;
//			label[i][0] = 2 * Math.pow(feature[i][0], 2) + 1;
			label[i][0] = 2 * feature[i][0] + 1;
		}

		double[] temp;
		int randomNumber;
		for (int i = 0; i < label.length; i++)
		{
			randomNumber = (int) (Math.random() * label.length);
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

		AbstractLoss loss = new MeanSquaredError();

		Model model = new Model();
		model.add(new Layer(5, new Random(), new Sigmoid()));
		model.add(new Layer(1, new Random(), new Relu()));
		model.compile(inputShape, loss, new BackPropagation(0.05));
		model.fit(trainFeature, trainLabel, 2000);
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
