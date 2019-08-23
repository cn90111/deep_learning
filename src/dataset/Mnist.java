package dataset;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class Mnist extends Dataset
{
	public Mnist() throws FileNotFoundException
	{
		super();
	}

	protected void arrayInit()
	{
		trainFeature = new double[60000][784];
		trainLabel = new double[trainFeature.length][10];
		testFeature = new double[10000][784];
		testLabel = new double[testFeature.length][10];
	}

	protected void loadData() throws FileNotFoundException
	{
		loadTrainX();
	}

	private void loadTrainX() throws FileNotFoundException
	{
		File file = new File("./dataset/mnist/train_x.txt");
		Scanner reader = new Scanner(file);

		int lineCount = 0;
		while (reader.hasNextLine())
		{
			String line = reader.nextLine();
			String[] tokens = line.split(",");

			for (int i = 0; i < trainFeature[lineCount].length; i++)
			{
				trainFeature[lineCount][i] = Double.parseDouble(tokens[i]);
			}
			lineCount++;
		}
		reader.close();
	}

	private void loadTrainY() throws FileNotFoundException
	{
		File file = new File("./dataset/mnist/train_y.txt");
		Scanner reader = new Scanner(file);

		int lineCount = 0;
		while (reader.hasNextLine())
		{
			String line = reader.nextLine();
			String[] tokens = line.split(",");

			for (int i = 0; i < trainLabel[lineCount].length; i++)
			{
				trainLabel[lineCount][i] = Double.parseDouble(tokens[i]);
			}
			lineCount++;
		}
		reader.close();
	}
	
	private void loadTestX() throws FileNotFoundException
	{
		File file = new File("./dataset/mnist/test_x.txt");
		Scanner reader = new Scanner(file);

		int lineCount = 0;
		while (reader.hasNextLine())
		{
			String line = reader.nextLine();
			String[] tokens = line.split(",");

			for (int i = 0; i < testFeature[lineCount].length; i++)
			{
				testFeature[lineCount][i] = Double.parseDouble(tokens[i]);
			}
			lineCount++;
		}
		reader.close();
	}

	private void loadTestY() throws FileNotFoundException
	{
		File file = new File("./dataset/mnist/test_y.txt");
		Scanner reader = new Scanner(file);

		int lineCount = 0;
		while (reader.hasNextLine())
		{
			String line = reader.nextLine();
			String[] tokens = line.split(",");

			for (int i = 0; i < testLabel[lineCount].length; i++)
			{
				testLabel[lineCount][i] = Double.parseDouble(tokens[i]);
			}
			lineCount++;
		}
		reader.close();
	}


	public double[][] getTrainFeature()
	{
		double[][] temp = new double[trainFeature.length][];

		for (int i = 0; i < temp.length; i++)
		{
			temp[i] = trainFeature[i].clone();
		}

		return temp;
	}

	public double[][] getTrainLabel()
	{
		double[][] temp = new double[trainLabel.length][];

		for (int i = 0; i < temp.length; i++)
		{
			temp[i] = trainLabel[i].clone();
		}

		return temp;
	}

	public double[][] getTestFeature()
	{
		double[][] temp = new double[testFeature.length][];

		for (int i = 0; i < temp.length; i++)
		{
			temp[i] = testFeature[i].clone();
		}

		return temp;
	}

	public double[][] getTestLabel()
	{
		double[][] temp = new double[testLabel.length][];

		for (int i = 0; i < temp.length; i++)
		{
			temp[i] = testLabel[i].clone();
		}

		return temp;
	}
}
