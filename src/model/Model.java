package model;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Scanner;

import activation.AbstractActivation;
import activation.Linear;
import activation.Relu;
import activation.Sigmoid;
import activation.Softmax;
import initializer.Zeros;
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

	public void setLayerArray(Layer[] layerArray)
	{
		this.layerArray = layerArray;

		layers.clear();
		for (int i = 0; i < layerArray.length; i++)
		{
			layers.add(layerArray[i]);
		}
	}

	public Layer[] getLayerArray()
	{
		Layer[] temp = new Layer[layerArray.length];
		for (int i = 0; i < layerArray.length; i++)
		{
			temp[i] = new Layer(layerArray[i]);
		}
		return temp;
	}

	public void summary()
	{
		double[][] weight;
		for (int i = 0; i < layerArray.length; i++)
		{
			System.out.println("layer : " + i);
			System.out.println("neuron size : " + layerArray[i].getNeuronSize());
			weight = layerArray[i].getWeight();
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
		neuronLink(inputFeatureSize);
		optimizer.setConfiguration(layerArray, loss);
	}

	private void neuronLink(int inputFeatureSize)
	{
		layerArray = layers.toArray(new Layer[0]);

		layerArray[0].setLinkSize(inputFeatureSize);
		for (int i = 1; i < layerArray.length; i++)
		{
			layerArray[i].setLinkSize(layerArray[i - 1].getNeuronSize());
		}
	}

	public void fit(double[][] feature, double[][] trueValue, int epochs)
	{
		if (layerArray == null || optimizer == null || loss == null)
		{
			System.out.println("need compile first.");
		}
		else
		{
			fit(feature, trueValue, epochs, true);
		}
	}

	public void fit(double[][] feature, double[][] trueValue, int epochs, boolean shuffle)
	{
		double[] guessValue;
		double error;
		for (int i = 0; i < epochs; i++)
		{
			error = 0;
			optimizer.newEpoch(i);

			// shuffle
			if (shuffle == true)
			{
				shuffle(feature, trueValue);
			}

			for (int j = 0; j < feature.length; j++)
			{
				guessValue = predict(feature[j]);
				optimizer.update(guessValue, trueValue[j]);
				// System.out.println(loss.getError(guessValue, trueValue[j]));
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
		if (layerArray == null)
		{
			System.out.println("need compile first.");
			return null;
		}

		layerArray[0].dataIn(feature);
		for (int i = 1; i < layerArray.length; i++)
		{
			layerArray[i].dataIn(layerArray[i - 1].dataOut());
		}
		return layerArray[layerArray.length - 1].dataOut();
	}

	public double[][][] getWeight()
	{
		double[][][] weight = new double[layerArray.length][][];
		for (int i = 0; i < layerArray.length; i++)
		{
			weight[i] = layerArray[i].getWeight();
		}
		return weight;
	}

	public double[][] getBias()
	{
		double[][] bias = new double[layerArray.length][];
		for (int i = 0; i < layerArray.length; i++)
		{
			bias[i] = layerArray[i].getBias();
		}
		return bias;
	}

	public void save(String filePath) throws IOException
	{
		FileWriter fw = new FileWriter(filePath);

		for (int i = 0; i < layerArray.length; i++)
		{
			fw.write("layer:" + i + ", ");
			fw.write("activation:" + layerArray[i].getActivation().toString() + ", ");
			fw.write("input size:" + layerArray[i].getLinkSize() + ", ");
			fw.write("output size:" + layerArray[i].getNeuronSize() + "\n");
			fw.flush();
		}

		for (int i = 0; i < layerArray.length; i++)
		{
			double[][] weight = layerArray[i].getWeight();
			double[] bias = layerArray[i].getBias();

			fw.write("weight[");
			for (int j = 0; j < weight.length; j++)
			{
				for (int k = 0; k < weight[j].length; k++)
				{
					fw.write(String.valueOf(weight[j][k]) + ", ");

					if ((j * weight[j].length + (k + 1)) % 5 == 0)
					{
						fw.write("\n");
					}
				}
			}
			fw.write("]" + "\n");
			fw.flush();

			fw.write("bias[");
			for (int j = 0; j < bias.length; j++)
			{
				fw.write(String.valueOf(bias[j]) + ", ");

				if ((j + 1) % 5 == 0)
				{
					fw.write("\n");
				}
			}
			fw.write("]" + "\n");
			fw.flush();
		}
		fw.close();
	}

	public static Model load(String filePath)
			throws FileNotFoundException, ClassNotFoundException, NoSuchMethodException, SecurityException,
			InstantiationException, IllegalAccessException, IllegalArgumentException, InvocationTargetException
	{
		Model loadModel = new Model();
		Layer[] loadLayer = null;
		Scanner weightFile = new Scanner(new File(filePath));

		AbstractActivation activation = null;

		String line;
		String[] tokens;
		String mode = null;
		String previousMode = null;

		int inputShape = 0;
		int outputShape = 0;

		int neuronCount = 0;
		int linkCount = 0;

		ArrayList<Double> value = new ArrayList<Double>();
		double[][] weight = null;
		double[] bias = null;

		int nowLayerCount = 0;

		boolean findSpecialToken;

		while (weightFile.hasNextLine())
		{
			line = weightFile.nextLine();
			tokens = line.split("[,:\\s\\[\\]]");
			value.clear();
			findSpecialToken = false;

			for (int i = 0; i < tokens.length; i++)
			{
				if (tokens[i].equals(""))
				{
					for (int j = i; j < tokens.length - 1; j++)
					{
						tokens[j] = tokens[j + 1];
						tokens[j + 1] = "";
					}
				}
			}

			for (int i = 0; i < tokens.length; i++)
			{
				System.out.println(tokens[i]);
				switch (tokens[i])
				{
					case "layer":
						mode = "add layer";
						findSpecialToken = true;
						break;
					case "activation":
						i++;
						switch (tokens[i])
						{
							case "Relu":
								activation = new Relu();
								break;
							case "Linear":
								activation = new Linear();
								break;
							case "Sigmoid":
								activation = new Sigmoid();
								break;
							case "Softmax":
								activation = new Softmax();
								break;
							default:
								System.out.println("Not support " + tokens[i] + "activation");
								System.exit(0);
						}
						findSpecialToken = true;
						break;
					case "input":
						i = i + 2;
						if (inputShape == 0)
						{
							inputShape = Integer.parseInt(tokens[i]);
						}
						findSpecialToken = true;
						break;
					case "output":
						i = i + 2;
						outputShape = Integer.parseInt(tokens[i]);
						findSpecialToken = true;
						break;
					case "weight":
						previousMode = mode;
						mode = "add weight";
						for (int j = i + 1; j < tokens.length; j++)
						{
							if (!tokens[j].equals(""))
							{
								value.add(Double.parseDouble(tokens[j]));
							}
							else
							{
								j = tokens.length;
							}
						}
						findSpecialToken = true;
						break;
					case "bias":
						previousMode = mode;
						mode = "add bias";
						for (int j = i + 1; j < tokens.length; j++)
						{
							if (!tokens[j].equals(""))
							{
								value.add(Double.parseDouble(tokens[j]));
							}
							else
							{
								j = tokens.length;
							}
						}
						findSpecialToken = true;
						break;
				}
			}

			if (findSpecialToken == false)
			{
				previousMode = mode;
				for (int i = 0; i < tokens.length; i++)
				{
					if (!tokens[i].equals(""))
					{
						value.add(Double.parseDouble(tokens[i]));
					}
					else
					{
						i = tokens.length;
					}
				}
			}

			switch (mode)
			{
				case "add layer":
					loadModel.add(new Layer(outputShape, new Zeros(), activation));
					break;
				case "add weight":
					if (previousMode.equals("add layer"))
					{
						loadModel.neuronLink(inputShape);
						loadLayer = loadModel.getLayerArray();
						nowLayerCount = 0;
						weight = loadLayer[nowLayerCount].getWeight();
						bias = loadLayer[nowLayerCount].getBias();
						neuronCount = 0;
						linkCount = 0;
					}
					else if (previousMode.equals("add bias"))
					{
						loadLayer[nowLayerCount].updateBias(bias);

						nowLayerCount = nowLayerCount + 1;
						weight = loadLayer[nowLayerCount].getWeight();
						bias = loadLayer[nowLayerCount].getBias();
						neuronCount = 0;
						linkCount = 0;
					}
					for (int i = 0; i < value.size(); i++)
					{
						weight[neuronCount][linkCount] = value.get(i);
						linkCount = linkCount + 1;
						if (linkCount == weight[neuronCount].length)
						{
							neuronCount = neuronCount + 1;
							linkCount = 0;
						}
					}
					break;
				case "add bias":
					if (previousMode.equals("add weight"))
					{
						neuronCount = 0;
						loadLayer[nowLayerCount].updateWeight(weight);
					}
					for (int i = 0; i < value.size(); i++)
					{
						bias[neuronCount] = value.get(i);
						neuronCount = neuronCount + 1;
					}
					break;
			}
		}
		loadLayer[nowLayerCount].updateBias(bias);
		loadModel.setLayerArray(loadLayer);
		weightFile.close();
		return loadModel;
	}

	private void shuffle(double[][] feature, double[][] trueValue)
	{
		double[] temp;
		int randomNumber;
		for (int i = 0; i < trueValue.length; i++)
		{
			randomNumber = (int) (Math.random() * trueValue.length);
			temp = feature[i];
			feature[i] = feature[randomNumber];
			feature[randomNumber] = temp;
			temp = trueValue[i];
			trueValue[i] = trueValue[randomNumber];
			trueValue[randomNumber] = temp;
		}

	}
}
