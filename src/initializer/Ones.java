package initializer;

public class Ones extends AbstractInitializer
{
	@Override
	public double[] initialize(int length)
	{
		for (int i = 0; i < length; i++)
		{
			result[i] = 1;
		}
		return result;
	}
}