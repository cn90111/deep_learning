package optimizer;

import activation.AbstractActivation;
import activation.Differentiable;
import layer.Layer;
import loss.AbstractLossFunction;

// batch bp 
// https://stats.stackexchange.com/questions/174708/how-are-weights-updated-in-the-batch-learning-method-in-neural-networks
public class BackPropagation extends Optimizer implements SupportBatchUpdate
{
	private double learningRate;
	private int batchSize;
	private int batchCount;
	// layer neurons weight
	private double[][][] weightDeltaArray;
	// layer bias
	private double[][] biasDeltaArray;
	public double updateCount;

	public BackPropagation(double learningRate)
	{
		this(learningRate, 1);
	}

	public BackPropagation(double learningRate, int batchSize)
	{
		this.learningRate = learningRate;
		this.batchSize = batchSize;
	}

	public void setConfiguration(Layer[] layers, AbstractLossFunction lossFunction)
	{
		this.layers = layers;
		this.lossFunction = lossFunction;
		this.weightDeltaArray = new double[layers.length][][];
		this.biasDeltaArray = new double[layers.length][];

		for (int i = 0; i < layers.length; i++)
		{
			weightDeltaArray[i] = layers[i].getWeight();
			biasDeltaArray[i] = layers[i].getBias();
		}
		resetBatch();
	}

	@Override
	public void resetBatch()
	{
		batchCount = 0;
		for (int i = 0; i < layers.length; i++)
		{
			for (int j = 0; j < weightDeltaArray[i].length; j++)
			{
				for (int k = 0; k < weightDeltaArray[i][j].length; k++)
				{
					weightDeltaArray[i][j][k] = 0;
				}
			}

			for (int j = 0; j < biasDeltaArray[i].length; j++)
			{
				biasDeltaArray[i][j] = 0;
			}
		}
	}

	@Override
	public void update(double[] guessValue, double[] trueValue)
	{
		double[] previousError;
		double[] nowError;

		batchCount = batchCount + 1;

		nowError = lossFunction.toDifferentiate(guessValue, trueValue);
		for (int i = layers.length - 1; i >= 0; i--)
		{
			previousError = activationBackPropagation(nowError, layers[i]);
			nowError = calculateError(previousError, layers[i].getWeight());
			calculateDeltaValue(previousError, layers[i].getInput(), i);
		}

		if (batchCount >= batchSize)
		{
			batchUpdate();
		}
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
		previousDataOutput = previousLayer.previousActivationOutput();
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

	private void calculateDeltaValue(double[] error, double[] previousDataOutput, int nowLayer)
	{
		for (int i = 0; i < error.length; i++)
		{
			for (int j = 0; j < previousDataOutput.length; j++)
			{
				weightDeltaArray[nowLayer][i][j] = weightDeltaArray[nowLayer][i][j]
						+ learningRate * error[i] * previousDataOutput[j];
			}
			biasDeltaArray[nowLayer][i] = biasDeltaArray[nowLayer][i] + learningRate * error[i];
		}
	}

	@Override
	public void batchUpdate()
	{
		for (int i = 0; i < layers.length; i++)
		{
			update(layers[i], i);
		}
		resetBatch();
	}

	private void update(Layer layer, int nowLayer)
	{
		double[][] weight = layer.getWeight();
		double[] bias = layer.getBias();
		for (int i = 0; i < weightDeltaArray[nowLayer].length; i++)
		{
			for (int j = 0; j < weightDeltaArray[nowLayer][i].length; j++)
			{
				weightDeltaArray[nowLayer][i][j] = weightDeltaArray[nowLayer][i][j] / batchCount;
				weight[i][j] = weight[i][j] - weightDeltaArray[nowLayer][i][j];
			}
		}

		for (int i = 0; i < biasDeltaArray[nowLayer].length; i++)
		{
			biasDeltaArray[nowLayer][i] = biasDeltaArray[nowLayer][i] / batchCount;
			bias[i] = bias[i] - biasDeltaArray[nowLayer][i];
		}
		layer.updateWeight(weight);
		layer.updateBias(bias);
	}

	@Override
	public int getBatchSize()
	{
		return batchSize;
	}

	@Override
	public void newEpoch(int currentEpoch)
	{

	}
}
