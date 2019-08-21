package optimizer;

import layer.Layer;
import loss.AbstractLossFunction;
import metaheuristic.Particle;

public class ParticleSwarmOptimization extends AdjustmentParticleSwarmOptimization
{
	public ParticleSwarmOptimization(metaheuristic.PsoParameter psoParameter, int dataSize)
	{
		super(psoParameter, dataSize);
	}

	@Override
	public void setConfiguration(Layer[] layers, AbstractLossFunction lossFunction)
	{
		super.setConfiguration(layers, lossFunction);
	}
	
	@Override
	public void update(double[] guessValue, double[] trueValue)
	{
		super.update(guessValue, trueValue);
	}
	
	@Override
	public void update()
	{
		if (firstEvalutate == true)
		{
			evaluate(featureArray, labelArray);
			determine();
			firstEvalutate = false;
		}
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
	}
	
	protected void evaluate(double[][] feature, double[][] label)
	{
		if (globalBestValue == 0)
		{
			setSolutionWeightToLayers(globalBestSolution);
			for (int i = 0; i < feature.length; i++)
			{
				globalBestValue = globalBestValue + evaluate(feature[i], label[i]);
			}
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
	}

	@Override
	public void newEpoch(int currentEpoch)
	{
		
	}
}
