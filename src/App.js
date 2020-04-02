import React from "react";
import * as ml5 from "ml5";
import styled from "styled-components";
import Sketch from "react-p5";

const App = () => {
  let video;
  let poseNet;
  let pose;
  let skeleton;

  let brain;
  let poseLabel = "";

  let state = "waiting";
  let targetLabel;

  const option = {
    architecture: "MobileNetV1",
    imageScaleFactor: 0.3,
    outputStride: 16,
    flipHorizontal: false,
    minConfidence: 0.75,
    maxPoseDetections: 1,
    scoreThreshold: 0.7,
    nmsRadius: 20,
    detectionType: "multiple",
    inputResolution: 257,
    multiplier: 0.5,
    quantBytes: 2
  };
  const setup = p5 => {
    p5.createCanvas(640, 640);
    video = p5.createCapture(p5.VIDEO);
    video.hide();

    poseNet = ml5.poseNet(video, option, modelLoaded);
    poseNet.on("pose", gotPoses);

    let options = {
      inputs: 34,
      outputs: 4,
      task: "classification",
      debug: true
    };

    brain = ml5.neuralNetwork(options);
    console.log(brain);
    // const modelInfo = {
    //   model: "model/model.json",
    //   metadata: "./model/model_meta.json",
    //   weights: "./model/model.weights.bin"
    // };
    // brain.load(modelInfo, brainLoaded);    // Error :  TypeError:Cannot read property 'inputMax' of undefined
    brain.loadData("./yaca.json", dataReady); // Error :   TypeError: Cannot read property 'save' of null
  };

  const dataReady = () => {
    brain.train(
      {
        epochs: 50 // 학습하는 거임 (50번 반복해서 학습한다.)
      },
      finished
    );
  };

  const finished = () => {
    console.log("model trained");
    brain.save();
    classifyPose();
  };

  const brainLoaded = () => {
    console.log("pose classification ready!");
    classifyPose();
  };

  const classifyPose = () => {
    if (pose) {
      let inputs = [];
      for (let i = 0; i < pose.keypoints.length; i++) {
        let x = pose.keypoints[i].position.x;
        let y = pose.keypoints[i].position.y;
        inputs.push(x);
        inputs.push(y);
      }

      brain.classify(inputs, gotResult);
    } else {
      setTimeout(classifyPose, 100);
    }
  };

  const gotResult = (error, results) => {
    if (results[0].confidence > 0.75) {
      poseLabel = results[0].label.toUpperCase(); // results[0].label 이 어떤 것인지를 나타냄 동작의 이름
    }
    classifyPose();
  };

  const gotPoses = poses => {
    if (poses.length > 0) {
      pose = poses[0].pose;
      skeleton = poses[0].skeleton;

      if (state === "collecting") {
        let inputs = [];
        for (let i = 0; i < pose.keypoints.length; i++) {
          let x = pose.keypoints[i].position.x;
          let y = pose.keypoints[i].position.y;
          inputs.push(x);
          inputs.push(y);
        }
        let target = [targetLabel];
        brain.addData(inputs, target);
      }
    }
  };

  const modelLoaded = () => {
    console.log("poseNet ready");
  };

  const draw = p5 => {
    p5.push();
    p5.translate(video.width, 0);
    p5.scale(-1, 1);
    p5.image(video, 0, 0, video.width, video.height);

    if (pose) {
      for (let i = 0; i < skeleton.length; i++) {
        let a = skeleton[i][0];
        let b = skeleton[i][1];

        p5.strokeWeight(2);
        p5.stroke(0);

        p5.line(a.position.x, a.position.y, b.position.x, b.position.y);
      }

      for (let i = 0; i < pose.keypoints.length; i++) {
        let x = pose.keypoints[i].position.x;
        let y = pose.keypoints[i].position.y;

        p5.fill(0);
        p5.stroke(255);
        p5.ellipse(x, y, 16, 16);
      }
    }

    p5.pop();

    p5.fill(255, 0, 255);
    p5.noStroke();
    p5.textSize(512);
    p5.textAlign(p5.CENTER, p5.CENTER);
    p5.text(poseLabel, p5.width / 2, p5.height / 2);
  };

  const buttons = {
    name: ["Stand", "Down", "Save", "Learn"],
    color: ["#e17055", "#fdcb6e", "#00cec9", "#0984e3"],
    hover: ["#fab1a0", "#ffeaa7", "#81ecec", "#74b9ff"]
  };

  const Button = styled.button`
    border: 1px;
    border-radius: 5px;
    background-color: ${props => props.color};

    width: 100px;
    height: 25px;

    color: white;

    margin-left: 5px;

    :hover {
      background-color: ${props => props.hover};
    }
  `;

  const ClickButton = e => {
    const poses = e.target.value;

    if (poses === "Save") {
      brain.saveData();
    } else if (poses === "Learn") {
      brain.normalizeData();
      brain.train({ epochs: 50 }, finished);
    } else {
      targetLabel = poses;
      console.log(targetLabel);

      setTimeout(function() {
        console.log("collecting");
        state = "collecting";
        setTimeout(function() {
          console.log("not Collecting");
          state = "waiting";
        }, 5000);
      }, 5000);
    }
  };

  return (
    <div>
      <Sketch setup={setup} draw={draw} />
      {buttons.name.map((button, index) => {
        return (
          <Button
            color={buttons.color[index]}
            hover={buttons.hover[index]}
            onClick={ClickButton}
            value={button}
            key={index}
          >
            {button}
          </Button>
        );
      })}
    </div>
  );
};

export default App;
