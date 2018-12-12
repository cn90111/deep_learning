package optimizer;

import layer.Layer;

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
	public double[][][] updateWeight(double error)
	{
		// TODO Auto-generated method stub
		return null;
	}
}
