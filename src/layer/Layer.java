package layer;

import activation.AbstractActivation;
import initializer.AbstractInitializer;

public abstract class Layer
{
	private Neurons[] neurons;
	private double[] bias;
	private AbstractInitializer biasInitializer;
	private AbstractActivation activation;

	private int linkSize;
	private double[] input;
	private double[] output;

	public Layer(int neuronSize, AbstractInitializer kernelInitializer, AbstractInitializer biasInitializer,
			AbstractActivation activation)
	{
		neurons = new Neurons[neuronSize];
		for (int i = 0; i < neurons.length; i++)
		{
			neurons[i] = new Neurons(kernelInitializer);
		}
		this.biasInitializer = biasInitializer;
		this.activation = activation;
	}

	public void dataIn(double[] data)
	{
		input = data;
		for (int i = 0; i < data.length; i++)
		{
			neurons[i].dataIn(data[i]);
		}
		output = calculate(input, bias, activation);
	}

	private double[] calculate(double[] input, double[] bias, AbstractActivation activation)
	{
		double[][] result = new double[neurons.length][];
		double[] output = new double[linkSize];

		for (int i = 0; i < output.length; i++)
		{
			output[i] = 0;
		}

		for (int i = 0; i < neurons.length; i++)
		{
			result[i] = neurons[i].dataOut();
		}

		for (int i = 0; i < output.length; i++)
		{
			for (int j = 0; j < neurons.length; j++)
			{
				output[i] = output[i] + result[j][i];
			}
		}

		for (int i = 0; i < output.length; i++)
		{
			output[i] = output[i] + bias[i];
		}

		output = activation.getError(output);

		return output;
	}

	public double[] dataOut()
	{
		return output;
	}

	public void setLinkSize(int size)
	{
		linkSize = size;
		for (Neurons neuron : neurons)
		{
			neuron.setLinkSize(linkSize);
		}
		bias = biasInitializer.initialize(linkSize);
	}

	public int getNeuronSize()
	{
		return neurons.length;
	}

	public void updateWeight(double[][] weight)
	{
		for (int i = 0; i < neurons.length; i++)
		{
			neurons[i].updateWeight(weight[i]);
		}
	}

	public double[][] getWeight()
	{
		double[][] weight = new double[neurons.length][];

		for (int i = 0; i < neurons.length; i++)
		{
			weight[i] = neurons[i].getWeight();
		}

		return weight;
	}
}
