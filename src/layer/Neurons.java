package layer;

import initializer.AbstractInitializer;

public class Neurons
{
	private int linkSize;
	private double input;
	private double[] weight;
	private double[] output;
	private AbstractInitializer initializer;

	public Neurons(AbstractInitializer initializer)
	{
		this.initializer = initializer;
	}

	public void setLinkSize(int size)
	{
		linkSize = size;
		weight = initializer.initialize(linkSize);
	}

	public void dataIn(double data)
	{
		input = data;
		output = calculate(input, weight);
	}

	private double[] calculate(double input, double[] weight)
	{
		double[] output = new double[linkSize];
		for (int i = 0; i < output.length; i++)
		{
			output[i] = input * weight[i];
		}
		return output;
	}

	public double[] dataOut()
	{
		return output;
	}

	public void updateWeight(double[] weight)
	{
		this.weight = weight;
	}

	public double[] getWeight()
	{
		return weight;
	}
}
