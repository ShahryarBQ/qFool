package FoolboxDemoGUI;

import javafx.collections.FXCollections;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ChoiceBox;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.VBox;
import javafx.scene.text.Text;

public class ModelPane extends GridPane {
    private final ChoiceBox<String> modelChoiceBox;
    private final ChoiceBox<String> vggChoiceBox;
    private final ChoiceBox<String> resNetChoiceBox;
    private final ChoiceBox<String> squeezeNetChoiceBox;
    private final ChoiceBox<String> denseNetChoiceBox;
    private final ChoiceBox<String> shuffleNetV2ChoiceBox;
    private final ChoiceBox<String> mobileNetChoiceBox;
    private final ChoiceBox<String> mobileNetV3ChoiceBox;
    private final ChoiceBox<String> resNeXtChoiceBox;
    private final ChoiceBox<String> wideResNetChoiceBox;
    private final ChoiceBox<String> mnasNetChoiceBox;

    private final CheckBox vggCheckBox;

    public ModelPane (GridPane gridPane) {
        super();
        this.setHgap(10);
        this.setVgap(10);

        // Texts
        Text modelText = new Text("Model: ");
        Text modelVersionText = new Text("Version: ");
        Text mobileNetV3Text = new Text("Size: ");

        // String Arrays
        String[] modelsList = new String[]{"AlexNet", "DenseNet", "GoogLeNet (Inception v1)",
                "Inception v3", "MNASNet", "MobileNet", "ResNet", "ResNeXt", "ShuffleNet V2",
                "SqueezeNet", "VGG", "Wide ResNet"};
        String[] vggList = new String[]{"11", "13", "16", "19"};
        String[] resNetList = new String[]{"18", "34", "50", "101", "152"};
        String[] squeezeNetList = new String[]{"1.0", "1.1"};
        String[] denseNetList = new String[]{"121", "161", "169", "201"};
        String[] shuffleNetV2List = new String[]{"x0.5", "x1.0", "x1.5", "x2.0"};
        String[] mobileNetList = new String[]{"V2", "V3"};
        String[] mobileNetV3List = new String[]{"Large", "Small"};
        String[] resNeXtList = new String[]{"50-32x4d", "101-32x8d"};
        String[] wideResNetList = new String[]{"50-2", "101-2"};
        String[] mnasNetList = new String[]{"0.5", "0.75", "1.0", "1.3"};

        // String
        String vggStr = "With Batch Normalization?";

        // ChoiceBoxes
        this.modelChoiceBox = new ChoiceBox<>(FXCollections.observableArrayList(modelsList));
        this.vggChoiceBox = new ChoiceBox<>(FXCollections.observableArrayList(vggList));
        this.resNetChoiceBox = new ChoiceBox<>(FXCollections.observableArrayList(resNetList));
        this.squeezeNetChoiceBox = new ChoiceBox<>(FXCollections.observableArrayList(squeezeNetList));
        this.denseNetChoiceBox = new ChoiceBox<>(FXCollections.observableArrayList(denseNetList));
        this.shuffleNetV2ChoiceBox = new ChoiceBox<>(FXCollections.observableArrayList(shuffleNetV2List));
        this.mobileNetChoiceBox = new ChoiceBox<>(FXCollections.observableArrayList(mobileNetList));
        this.mobileNetV3ChoiceBox = new ChoiceBox<>(FXCollections.observableArrayList(mobileNetV3List));
        this.resNeXtChoiceBox = new ChoiceBox<>(FXCollections.observableArrayList(resNeXtList));
        this.wideResNetChoiceBox = new ChoiceBox<>(FXCollections.observableArrayList(wideResNetList));
        this.mnasNetChoiceBox = new ChoiceBox<>(FXCollections.observableArrayList(mnasNetList));

        // CheckBox
        this.vggCheckBox = new CheckBox(vggStr);

        // modelOptions
        final GridPane[] modelOptionsGridPane = {new GridPane()};
        final VBox[] modelOptions2VBox = {new VBox()};
        modelOptionsGridPane[0].add(modelOptions2VBox[0], 0, 1);

        // ModelPane
        VBox modelVBox = new VBox(modelText, this.modelChoiceBox);
        this.add(modelVBox, 0, 0);
        this.add(modelOptionsGridPane[0], 0, 1);

        // Event Handlers
        this.modelChoiceBox.setOnAction(e -> {
            gridPane.getChildren().remove(modelOptionsGridPane[0]);
            modelOptionsGridPane[0] = new GridPane();

            String selectedModel = this.modelChoiceBox.getValue();
            VBox modelOptionsVBox = new VBox();
            switch (selectedModel) {
                case "VGG" ->
                        modelOptionsVBox = new VBox(modelVersionText, this.vggChoiceBox, this.vggCheckBox);
                case "ResNet" ->
                        modelOptionsVBox = new VBox(modelVersionText, this.resNetChoiceBox);
                case "SqueezeNet" ->
                        modelOptionsVBox = new VBox(modelVersionText, this.squeezeNetChoiceBox);
                case "DenseNet" ->
                        modelOptionsVBox = new VBox(modelVersionText, this.denseNetChoiceBox);
                case "ShuffleNet V2" ->
                        modelOptionsVBox = new VBox(modelVersionText, this.shuffleNetV2ChoiceBox);
                case "MobileNet" ->
                        modelOptionsVBox = new VBox(modelVersionText, this.mobileNetChoiceBox);
                case "ResNeXt" ->
                        modelOptionsVBox = new VBox(modelVersionText, this.resNeXtChoiceBox);
                case "Wide ResNet" ->
                        modelOptionsVBox = new VBox(modelVersionText, this.wideResNetChoiceBox);
                case "MNASNet" ->
                        modelOptionsVBox = new VBox(modelVersionText, this.mnasNetChoiceBox);
            }

            modelOptionsGridPane[0].add(modelOptionsVBox, 0, 0);
            gridPane.add(modelOptionsGridPane[0], 1, 1);
        });

        this.mobileNetChoiceBox.setOnAction(e -> {
            modelOptionsGridPane[0].getChildren().remove(modelOptions2VBox[0]);
            modelOptions2VBox[0] = new VBox();

            String selectedMobileNet = this.mobileNetChoiceBox.getValue();
            if (selectedMobileNet.equals("V3"))
                modelOptions2VBox[0] = new VBox(mobileNetV3Text, this.mobileNetV3ChoiceBox);

            modelOptionsGridPane[0].add(modelOptions2VBox[0], 0, 1);
        });
    }

    public String determineModel() {
        String modelStr = this.modelChoiceBox.getValue();
        String postfix = "";
        switch (modelStr) {
            case "VGG" -> {
                postfix = this.vggChoiceBox.getValue();
                if (this.vggCheckBox.isSelected())
                    postfix = postfix + "bn";
            }
            case "ResNet" -> postfix = this.resNetChoiceBox.getValue();
            case "SqueezeNet" -> postfix = this.squeezeNetChoiceBox.getValue();
            case "DenseNet" -> postfix = this.denseNetChoiceBox.getValue();
            case "ShuffleNet V2" -> postfix = this.shuffleNetV2ChoiceBox.getValue();
            case "MobileNet" -> postfix = this.mobileNetChoiceBox.getValue() + this.mobileNetV3ChoiceBox.getValue();
            case "ResNeXt" -> postfix = this.resNeXtChoiceBox.getValue();
            case "Wide ResNet" -> postfix = this.wideResNetChoiceBox.getValue();
            case "MNASNet" -> postfix = this.mnasNetChoiceBox.getValue();
        }
        return modelStr + postfix;
    }
}
