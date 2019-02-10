package initializer;

public class Random extends AbstractInitializer
{
	private double upperLimit;
	private double lowerLimit;

	public Random(double upperLimit, double lowerLimit)
	{
		this.upperLimit = upperLimit;
		this.lowerLimit = lowerLimit;
	}

	@Override
	public double[] initialize(int length)
	{
		result = new double[length];
		for (int i = 0; i < length; i++)
		{
			result[i] = Math.random() * (upperLimit - lowerLimit) + lowerLimit; // upperLimit ~ lowerLimit
		}
		return result;
	}
}
