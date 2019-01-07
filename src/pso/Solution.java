package pso;

public class Solution
{
	private double[][][] weight;
	private double[][] bias;

	public Solution()
	{

	}

	public Solution(double[][][] weight, double[][] bias)
	{
		this.weight = weight;
		this.bias = bias;
	}

	public Solution(Solution other)
	{
		this.weight = other.getWeight();
		this.bias = other.getBias();
	}

	public double[][][] getWeight()
	{
		double[][][] temp = new double[weight.length][][];
		for (int i = 0; i < temp.length; i++)
		{
			temp[i] = new double[weight[i].length][];
			for (int j = 0; j < temp[i].length; j++)
			{
				temp[i][j] = new double[weight[i][j].length];
				for (int k = 0; k < temp[i][j].length; k++)
				{
					temp[i][j][k] = weight[i][j][k];
				}
			}
		}
		return temp;
	}

	public void setWeight(double[][][] weight)
	{
		this.weight = weight;
	}

	public double[][] getBias()
	{
		double[][] temp = new double[bias.length][];
		for (int i = 0; i < temp.length; i++)
		{
			temp[i] = new double[bias[i].length];
			for (int j = 0; j < temp[i].length; j++)
			{
				temp[i][j] = bias[i][j];
			}
		}
		return temp;
	}

	public void setBias(double[][] bias)
	{
		this.bias = bias;
	}
}
