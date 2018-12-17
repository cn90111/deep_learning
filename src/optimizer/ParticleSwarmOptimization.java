package optimizer;

import layer.Layer;
import loss.AbstractLoss;

public class ParticleSwarmOptimization extends AbstractOptimizer
{
	private int particleSize;
	private double c1;
	private double c2;
	private double[][][] globalBestSolution;

	public ParticleSwarmOptimization(int particleSize, double globalSearchWeight, double localSearchWeight)
	{
		this.particleSize = particleSize;
		this.c1 = localSearchWeight;
		this.c2 = globalSearchWeight;
	}

	@Override
	public void setLayers(Layer[] layers)
	{
		globalBestSolution = new double[layers.length][][];
		for (int i = 0; i < layers.length; i++)
		{
			globalBestSolution[i] = layers[i].getWeight();
		}
	}

	@Override
	public void update(AbstractLoss loss, double guessValue[], double trueValue[])
	{
		// TODO Auto-generated method stub
		return null;
	}
}
