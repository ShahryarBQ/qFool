package FoolboxDemoGUI;

import javafx.application.Application;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.layout.GridPane;
import javafx.stage.Stage;

public class GUI extends Application {

    @Override
    public void start(Stage primaryStage) {
        // create the main GridPane
        GridPane gridPane = new GridPane();
        gridPane.setAlignment(Pos.CENTER);
        gridPane.setHgap(10);
        gridPane.setVgap(10);

        // create sub-Panes which make up the main gridPane
        AttackPane attackPane = new AttackPane(gridPane);
        ModelPane modelPane = new ModelPane(gridPane);
        DatasetPane datasetPane = new DatasetPane();
        InputImagePane inputImagePane = new InputImagePane(primaryStage);
        ImagePane outputImagePane = new ImagePane("Output Image: \n(Adversarial Example)");
        ImagePane perturbationImagePane = new ImagePane("Perturbation Image: ");
        NormPane normPane = new NormPane();
        RunPane runPane = new RunPane(attackPane, modelPane, datasetPane, inputImagePane,
                outputImagePane, perturbationImagePane, normPane);

        // add each sub-Pane to the main gridPane
            // Row 0 & 1
        gridPane.add(attackPane, 0, 0, 1, 1);
        gridPane.add(modelPane, 1, 0, 1, 1);
        gridPane.add(datasetPane, 2, 0, 1, 1);
        gridPane.add(inputImagePane, 3, 0, 1, 2);
        gridPane.add(outputImagePane, 4, 0, 1, 2);
        gridPane.add(perturbationImagePane, 5, 0, 1, 2);
            // Row 2
        gridPane.add(normPane, 0, 2, 1, 1);
        gridPane.add(runPane, 5, 2, 1, 1);

        // Scene
        Scene scene = new Scene(gridPane);

        // Stage
        primaryStage.setScene(scene);
        primaryStage.setTitle("Foolbox Demo");
        primaryStage.setMinWidth(1024);
        primaryStage.setMinHeight(384);
        primaryStage.show();
    }

    public static void main(String[] args) {
        Application.launch(args);
    }
}
