package optimizer;

import activation.AbstractActivation;
import activation.Differentiable;
import layer.Layer;
import loss.AbstractLoss;

public class BackPropagation extends AbstractOptimizer
{
	private double learningRate;
	private Layer[] layers;

	public BackPropagation(double learningRate)
	{
		this.learningRate = learningRate;
	}

	public void setLayers(Layer[] layers)
	{
		this.layers = layers;
	}

	@Override
	public void update(AbstractLoss loss, double guessValue[], double trueValue[])
	{
		double[] previousError;
		double[] nowError;

		nowError = loss.toDifferentiate(guessValue, trueValue);
		for (int i = layers.length - 1; i > 0; i--)
		{
			previousError = activationBackPropagation(nowError, layers[i]);
			nowError = calculateError(previousError, layers[i].getWeight());
			update(layers[i], previousError, layers[i - 1].dataOut());
		}
	}

	private void update(Layer layer, double[] error, double[] previousDataOutput)
	{
		double[][] weight = layer.getWeight();
		double[] bias = layer.getBias();
		for (int i = 0; i < error.length; i++)
		{
			for (int j = 0; j < previousDataOutput.length; j++)
			{
				weight[i][j] = weight[i][j] - learningRate * error[i] * previousDataOutput[j];
				bias[i] = bias[i] - learningRate * error[i];
			}
		}
		layer.updateWeight(weight);
		layer.updateBias(bias);
	}

	private double[] activationBackPropagation(double[] nowError, Layer previousLayer)
	{
		double[] previousDataOutput;
		double[] previousError = null;
		AbstractActivation previousActivation;

		previousActivation = previousLayer.getActivation();
		if (!(previousActivation instanceof Differentiable))
		{
			System.out.println("Activation can't Differential");
			System.exit(1);
		}
		previousDataOutput = previousLayer.dataOut();
		previousError = ((Differentiable) previousActivation).toDifferentiate(previousDataOutput);
		for (int i = 0; i < previousError.length; i++)
		{
			previousError[i] = previousError[i] * nowError[i];
		}

		return previousError;
	}

	private double[] calculateError(double[] error, double[][] weight)
	{
		double[] errorValue = new double[weight[0].length];
		for (int i = 0; i < errorValue.length; i++)
		{
			errorValue[i] = 0;
			for (int j = 0; j < error.length; j++)
			{
				errorValue[i] = errorValue[i] + error[j] * weight[j][i];
			}
		}
		return errorValue;
	}
}
