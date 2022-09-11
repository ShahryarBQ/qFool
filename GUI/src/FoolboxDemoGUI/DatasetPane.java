package FoolboxDemoGUI;

import javafx.collections.FXCollections;
import javafx.scene.control.ChoiceBox;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.VBox;
import javafx.scene.text.Text;

public class DatasetPane extends GridPane {
    private String datasetStr;

    public DatasetPane() {
        super();
        this.setHgap(10);
        this.setVgap(10);

        // Text
        Text datasetText = new Text("Dataset: ");

        // String Array
        String[] datasetsList = new String[]{"ImageNet"};

        // ChoiceBox
        ChoiceBox<String> datasetChoiceBox = new ChoiceBox<>(FXCollections.observableArrayList(datasetsList));

        // DatasetPane
        VBox datasetVBox = new VBox(datasetText, datasetChoiceBox);
        this.add(datasetVBox, 0, 0);

        // Event Handler
        datasetChoiceBox.setOnAction(e -> datasetStr = datasetChoiceBox.getValue());
    }

    public String determineDataset() { return datasetStr; }
}
