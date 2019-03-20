import java.io.FileWriter;
import java.io.IOException;

import loss.AbstractLossFunction;
import model.Model;

public class Timer extends Thread
{
	boolean stop = false;
	long timeCount = 0;

	Model model;
	double[][] feature;
	double[][] label;
	AbstractLossFunction lossFunction;
	FileWriter file;

	public Timer(Model model, double[][] feature, double[][] label, AbstractLossFunction lossFunction)
	{
		this.model = model;
		this.feature = feature;
		this.label = label;
		this.lossFunction = lossFunction;
	}

	@Override
	public void run()
	{
		double[][] predictLabel = new double[label.length][label[0].length];
		try
		{
			file = new FileWriter("result/timeMse.csv", true);
		}
		catch (IOException e1)
		{
			e1.printStackTrace();
		}
		while (!stop)
		{
			double lossValue = 0;
			double avgLossValue = 0;

			for (int j = 0; j < feature.length; j++)
			{
				predictLabel[j] = model.predict(feature[j]);
			}
			for (int j = 0; j < predictLabel.length; j++)
			{
				lossValue = lossFunction.getError(predictLabel[j], label[j]);
				avgLossValue = avgLossValue + lossValue;
			}

			avgLossValue = avgLossValue / predictLabel.length;

			try
			{
				file.write(timeCount + "," + avgLossValue + "\n");
			}
			catch (IOException e1)
			{
				e1.printStackTrace();
			}

			// System.out.println(timeCount + "s,mse:" + avgLossValue);

			try
			{
				Thread.sleep(100);
			}
			catch (InterruptedException e)
			{
				e.printStackTrace();
			}
			timeCount = timeCount + 1;
		}
	}

	public void close()
	{
		stop = true;

		try
		{
			file.close();
		}
		catch (IOException e)
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
