package FoolboxDemoGUI;

import javafx.scene.control.Button;
import javafx.scene.image.Image;
import javafx.scene.layout.GridPane;
import javafx.scene.text.TextAlignment;

import java.io.*;
import java.util.ArrayList;

public class RunPane extends GridPane {
    public RunPane(AttackPane attackPane,
                   ModelPane modelPane,
                   DatasetPane datasetPane,
                   InputImagePane inputImagePane,
                   ImagePane outputImagePane,
                   ImagePane perturbationImagePane,
                   NormPane normPane) {
        super();
        this.setHgap(10);
        this.setVgap(10);

        // Strings
        String runString = "Run\nSimulation";
        String python3command = "python3";
        String pythonFileAddress = System.getProperty("user.dir") + File.separator + "src" +
                File.separator + "FoolboxDemoLogic" + File.separator + "foolbox_demo.py";
        String outputImageAddress = "File:" + System.getProperty("user.dir") + File.separator +
                "misc" + File.separator + "misc_temp" + File.separator + "adv.jpg";
        String perturbationImageAddress = "File:" + System.getProperty("user.dir") + File.separator +
                "misc" + File.separator + "misc_temp" + File.separator + "pert.jpg";

        // Button
        Button runButton = new Button(runString);
        runButton.setTextAlignment(TextAlignment.CENTER);

        // RunPane
        this.add(runButton, 0, 0);

        // Event Handler
        runButton.setOnAction(e -> {
            Runnable runnable = () -> {
                runButton.setDisable(true);

                String attack = this.reformatString(attackPane.determineAttack());
                String model = this.reformatString(modelPane.determineModel());
                String dataset = this.reformatString(datasetPane.determineDataset());
                String inputImageAddress = inputImagePane.getInputImageAddress().substring(5);

                System.out.println(attack);
                System.out.println(model);
                System.out.println(dataset);
                System.out.println(inputImageAddress);

                ProcessBuilder processBuilder = new ProcessBuilder(python3command,
                        pythonFileAddress, attack, model, dataset, inputImageAddress);
                processBuilder.redirectErrorStream(true);
                Process process;
                ArrayList<String> lines = new ArrayList<>();
                try {
                    process = processBuilder.start();
                    BufferedReader results = new BufferedReader(new InputStreamReader(process.getInputStream()));
                    String line;
                    while ((line = results.readLine()) != null) {
                        String indicator = line.substring(0,4);
                        if (indicator.equals("mse:") || indicator.equals("ori:") || indicator.equals("adv:"))
                            lines.add(line.substring(4));
                    }
                    int exitCode = process.waitFor();
                } catch (IOException | InterruptedException ioException) {
                    ioException.printStackTrace();
                }

                Image outputImage = new Image(outputImageAddress);
                Image pertImage = new Image(perturbationImageAddress);
                outputImagePane.setImage(outputImage);
                perturbationImagePane.setImage(pertImage);

                normPane.setText(lines.get(0));
                inputImagePane.setLabelText(lines.get(1));
                outputImagePane.setLabelText(lines.get(2));

                runButton.setDisable(false);
            };

            runnable.run();
        });
    }

    private String reformatString(String s) {
        String newS = s.replace("(", "");
        newS = newS.replace(")", "");
        newS = newS.replace(".", "_");
        newS = newS.replace("-", "_");
        newS = newS.replace(" ", "");
        return newS;
    }
}
