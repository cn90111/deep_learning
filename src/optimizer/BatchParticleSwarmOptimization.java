package optimizer;

import layer.Layer;
import loss.AbstractLossFunction;
import pso.Particle;

public class BatchParticleSwarmOptimization extends AdjustmentParticleSwarmOptimization implements SupportBatchUpdate
{
	protected int batchSize;
	protected int batchCount;

	protected double[][] featureArray;
	protected double[][] labelArray;

	public BatchParticleSwarmOptimization(pso.Parameter psoParameter)
	{
		this(psoParameter, 1);
	}

	public BatchParticleSwarmOptimization(pso.Parameter psoParameter, int batch)
	{
		super(psoParameter);

		this.batchSize = batch;

		featureArray = new double[batchSize][];
		labelArray = new double[batchSize][];

		batchCount = 0;
	}

	@Override
	public void setConfiguration(Layer[] layers, AbstractLossFunction lossFunction)
	{
		super.setConfiguration(layers, lossFunction);

		for (int i = 0; i < batchSize; i++)
		{
			featureArray[i] = new double[layers[0].getLinkSize()];
			labelArray[i] = new double[layers[layers.length - 1].getNeuronSize()];
		}
	}

	protected void evaluate(double[][] feature, double[][] label)
	{
		// evaluate globalBestSolution loss with other feature input
		setSolutionWeightToLayers(globalBestSolution);
		globalBestValue = 0;
		for (int i = 0; i < feature.length; i++)
		{
			globalBestValue = globalBestValue + evaluate(feature[i], label[i]);
		}

		for (int i = 0; i < particle.length; i++)
		{
			evaluate(particle[i], feature, label);
		}
	}

	private void evaluate(Particle particle, double[][] feature, double[][] label)
	{
		double error = 0;

		setSolutionWeightToLayers(particle.getNowSolution());
		for (int i = 0; i < feature.length; i++)
		{
			error = error + evaluate(feature[i], label[i]);
		}
		particle.setNowValue(error);
	}

	@Override
	public void update(double[] guessValue, double[] trueValue)
	{
		updateCount = updateCount + 1;
		if (updateCount < pso.linearEndCount)
		{
			w = pso.maxW - (pso.linearEndW / pso.linearEndCount) * updateCount;
		}
		else
		{
			w = (pso.maxW - pso.linearEndW) * Math.exp((pso.linearEndCount - updateCount) / pso.nonlinearlyWeight);
		}
		double[] feature = layers[0].getInput();

		saveValueToArray(featureArray, feature, batchCount);
		saveValueToArray(labelArray, trueValue, batchCount);

		batchCount = batchCount + 1;

		if (batchCount >= batchSize)
		{
			batchUpdate();
		}
	}

	protected void saveValueToArray(double[][] array, double[] value, int index)
	{
		for (int i = 0; i < value.length; i++)
		{
			array[index][i] = value[i];
		}
	}

	@Override
	public void batchUpdate()
	{
		evaluate(featureArray, labelArray);
		determine();
		transit();
		evaluate(featureArray, labelArray);
		determine();

		double[][][] weight = globalBestSolution.getWeight();
		double[][] bias = globalBestSolution.getBias();
		for (int i = 0; i < layers.length; i++)
		{
			layers[i].updateWeight(weight[i]);
			layers[i].updateBias(bias[i]);
		}

		resetBatch();
	}

	@Override
	public int getBatchSize()
	{
		return batchSize;
	}

	@Override
	public void resetBatch()
	{
		batchCount = 0;

		for (int i = 0; i < featureArray.length; i++)
		{
			for (int j = 0; j < featureArray[i].length; j++)
			{
				featureArray[i][j] = 0;
			}

			for (int j = 0; j < labelArray[i].length; j++)
			{
				labelArray[i][j] = 0;
			}
		}
	}

	@Override
	public void newEpoch(int currentEpoch)
	{

	}
}
