package model;

import java.util.ArrayList;

import layer.Layer;
import loss.AbstractLossFunction;
import optimizer.Optimizer;
import optimizer.SupportBatchUpdate;

public class Model
{
	private ArrayList<Layer> layers = new ArrayList<Layer>();
	private Layer[] layerArray;
	private AbstractLossFunction loss;
	private Optimizer optimizer;

	public Model()
	{
	}

	public void add(Layer layer)
	{
		layers.add(layer);
	}

	public void summary()
	{
		double[][] weight;
		for (int i = 0; i < layers.size(); i++)
		{
			System.out.println("layer : " + i);
			System.out.println("neuron size : " + layers.get(i).getNeuronSize());
			weight = layers.get(i).getWeight();
			for (int j = 0; j < weight.length; i++)
			{
				for (int k = 0; k < weight[0].length; k++)
				{
					System.out.print(weight[j][k] + " ");
				}
				System.out.println("============");
			}
		}
	}

	public void compile(int inputFeatureSize, AbstractLossFunction loss, Optimizer optimizer)
	{
		this.loss = loss;
		this.optimizer = optimizer;

		layerArray = layers.toArray(new Layer[0]);

		layerArray[0].setLinkSize(inputFeatureSize);
		for (int i = 1; i < layerArray.length; i++)
		{
			layerArray[i].setLinkSize(layerArray[i - 1].getNeuronSize());
		}

		optimizer.setConfiguration(layerArray, loss);
	}

	public void fit(double[][] feature, double[][] trueValue, int epochs)
	{
		double[] guessValue = new double[feature.length];
		double error;
		for (int i = 0; i < epochs; i++)
		{
			error = 0;
			optimizer.newEpoch(i);
			for (int j = 0; j < feature.length; j++)
			{
				guessValue = predict(feature[j]);
				optimizer.update(guessValue, trueValue[j]);
				error = error + loss.getError(guessValue, trueValue[j]);
			}
			if (optimizer instanceof SupportBatchUpdate)
			{
				if (feature.length % ((SupportBatchUpdate) optimizer).getBatchSize() != 0)
				{
					((SupportBatchUpdate) optimizer).batchUpdate();
				}
			}
			// System.out.println(i + "th mse:" + error / feature.length);
		}
	}

	public double[] predict(double[] feature)
	{
		layerArray[0].dataIn(feature);
		for (int i = 1; i < layerArray.length; i++)
		{
			layerArray[i].dataIn(layerArray[i - 1].dataOut());
		}
		return layerArray[layerArray.length - 1].dataOut();
	}

	public double[][][] getWeight()
	{
		double[][][] weight = new double[layers.size()][][];
		for (int i = 0; i < layerArray.length; i++)
		{
			weight[i] = layerArray[i].getWeight();
		}
		return weight;
	}

	public double[][] getBias()
	{
		double[][] bias = new double[layers.size()][];
		for (int i = 0; i < layerArray.length; i++)
		{
			bias[i] = layerArray[i].getBias();
		}
		return bias;
	}
}
