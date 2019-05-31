package optimizer;

import layer.Layer;
import loss.AbstractLossFunction;
import metaheuristic.Particle;

public class BatchParticleSwarmOptimizationBackPropagation extends HybridParticleSwarmOptimizationBackPropagation
		implements SupportBatchUpdate
{

	public BatchParticleSwarmOptimizationBackPropagation(metaheuristic.PsoParameter psoParameter, int batch, int condition,
			int bpSearchGenerations, int psoGenerations, double learningRate, double learningRateDecayRate)
	{
		super(psoParameter, batch, condition, bpSearchGenerations, psoGenerations, learningRate, learningRateDecayRate);
	}

	@Override
	public void setConfiguration(Layer[] layers, AbstractLossFunction lossFunction)
	{
		super.setConfiguration(layers, lossFunction);

		resetBatch();
	}

	@Override
	public void update(double[] guessValue, double[] trueValue)
	{
		super.update(guessValue, trueValue);
	}

	@Override
	public void update()
	{
		batchUpdate();
	}

	@Override
	public void batchUpdate()
	{
		super.update();
		resetBatch();
	}

	@Override
	protected void psoUpdate()
	{
		evaluate(featureArray, labelArray);
		determine();
		super.psoUpdate();
	}

	@Override
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

		super.reset();
	}

	@Override
	public int getBatchSize()
	{
		return dataSize;
	}
}
