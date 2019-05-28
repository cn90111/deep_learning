package optimizer;

import layer.Layer;
import loss.AbstractLossFunction;
import pso.Particle;

public class BatchParticleSwarmOptimization extends AdjustmentParticleSwarmOptimization implements SupportBatchUpdate
{
	public BatchParticleSwarmOptimization(pso.Parameter psoParameter)
	{
		this(psoParameter, 1);
	}

	public BatchParticleSwarmOptimization(pso.Parameter psoParameter, int batch)
	{
		super(psoParameter, batch);
	}

	@Override
	public void setConfiguration(Layer[] layers, AbstractLossFunction lossFunction)
	{
		super.setConfiguration(layers, lossFunction);

		resetBatch();
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
		double lossValue = 0;

		setSolutionWeightToLayers(particle.getNowSolution());
		for (int i = 0; i < feature.length; i++)
		{
			lossValue = lossValue + evaluate(feature[i], label[i]);
		}
		particle.setNowValue(lossValue);

		lossValue = 0;
		setSolutionWeightToLayers(particle.getLocalBestSolution());
		for (int i = 0; i < feature.length; i++)
		{
			lossValue = lossValue + evaluate(feature[i], label[i]);
		}
		particle.setLocalBestValue(lossValue);
	}

	@Override
	public void update(double[] guessValue, double[] trueValue)
	{
		super.update(guessValue, trueValue);
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
		return dataSize;
	}

	@Override
	public void resetBatch()
	{
		dataCount = 0;

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

	@Override
	public void update()
	{
		batchUpdate();
	}
}
