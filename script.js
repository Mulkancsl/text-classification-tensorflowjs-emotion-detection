let input = document.querySelector("input");
let button = document.querySelector("button");
button.addEventListener("click", onClick);

var isModelLoaded = false;
var tesInit = "false";
let model;
let word2index;

const maxlen = 20;
const vocab_size = 2000;
const padding = "post";
const truncating = "post";

const app_url = "http://127.0.0.1:5500";

var myVar;

function myFunction() {
  myVar = setTimeout(showPage(), 3000);
}

function showPage() {
  document.getElementById("loaderlabel").style.display = "none";
  document.getElementById("loader").style.display = "none";
  document.getElementById("mainAPP").style.display = "block";
  if (!document.getElementById("input").value == "") {
    document.getElementById("recom1").style.display = "none";
    document.getElementById("recomLabel").style.display = "none";
  }
}

function detectWebGLContext() {
  var canvas = document.createElement("canvas");
  var gl =
    canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
  if (gl && gl instanceof WebGLRenderingContext) {
    console.log("Congratulations! Your browser supports WebGL.");
    init();
  } else {
    alert(
      "Failed to get WebGL context. Your browser or device may not support WebGL."
    );
  }
}

detectWebGLContext();

function getInput() {
  const reviewText = document.getElementById("input");
  return reviewText.value;
}

function padSequence(
  sequences,
  maxLen,
  padding = "post",
  truncating = "post",
  pad_value = 0
) {
  return sequences.map((seq) => {
    if (seq.length > maxLen) {
      if (truncating === "pre") {
        seq.splice(0, seq.length - maxLen);
      } else {
        seq.splice(maxLen, seq.length - maxLen);
      }
    }

    if (seq.length < maxLen) {
      const pad = [];
      for (let i = 0; i < maxLen - seq.length; i++) {
        pad.push(pad_value);
      }
      if (padding === "pre") {
        seq = pad.concat(seq);
      } else {
        seq = seq.concat(pad);
      }
    }
    return seq;
  });
}

function predict(inputText) {
  const sequence = inputText.map((word) => {
    let indexed = word2index[word];

    if (indexed === undefined) {
      return 1;
    }
    return indexed;
  });

  const paddedSequence = padSequence([sequence], maxlen);

  const score = tf.tidy(() => {
    const input = tf.tensor2d(paddedSequence, [1, maxlen]);
    const result = model.predict(input);
    return result.dataSync();
  });

  return score;
}

function onClick() {
  if (!isModelLoaded) {
    alert("Model not loaded yet");
    return;
  }

  if (getInput() === "") {
    alert("Review Can't be Null");
    document.getElementById("input").focus();
    return;
  }

  const inputText = getInput().trim().toLowerCase().split(" ");

  document.getElementById("recom1").style.display = "none";
  document.getElementById("recomLabel").style.display = "none";

  let score = predict(inputText);
  let max = -Infinity;

  let result = null;

  console.log(score);

  for (let i = 0; i < score.length; i++) {
    if (score[i] > max) {
      max = score[i];
      result = i;
    }
  }

  document.getElementById("resultbox").style.display = "block";
  document.getElementById("loaderResult").style.display = "block";
  document.getElementById("result").style.display = "none";
  document.getElementById("logsbox").style.display = "none";

  let label = ["marah", "takut", "senang", "cinta", "sedih", "terkejut"];

  setTimeout(() => {
    document.getElementById("loaderResult").style.display = "none";
    document.getElementById("result").style.display = "block";
    for (let i = 0; i < score.length; i++) {
      document.getElementById(i).innerText = score[i].toFixed(50);
    }

    for (let i = 0; i < label.length; i++) {
      if (result == i) {
        document.getElementById("result").innerText =
          "Dari pesan tersebut emosinya adalah " +
          label[i][0].toUpperCase() +
          label[i].slice(1);
      }
    }
  }, 500);
}

// =====================================================
const element = document.getElementById("toggleLogs");
const elementTarget = document.getElementById("logsbox");
element.addEventListener("click", function () {
  if (
    elementTarget.style.display === "none" ||
    elementTarget.style.display === ""
  ) {
    elementTarget.style.display = "block";
    element.innerText = "▲";
  } else {
    elementTarget.style.display = "none";
    element.innerText = "▼";
  }
});

document.getElementById("recom1").addEventListener("click", function (e) {
  const recom = e.target.innerText;
  document.getElementById("input").value = recom;
  e.target.style.display = "none";
  document.getElementById("recomLabel").style.display = "none";
});

document.getElementById("input").addEventListener("input", function (e) {
  if (e.target.value == "") {
    document.getElementById("recom1").style.display = "block";
    document.getElementById("recomLabel").style.display = "block";
  }
});
// =====================================================

async function init() {
  // Memanggil model tfjs
  model = await tf.loadLayersModel(
    "http://127.0.0.1:5500/tfjs_model/model.json"
  );
  isModelLoaded = true;

  //Memanggil word_index
  const word_indexjson = await fetch("http://127.0.0.1:5500/word_index.json");
  word2index = await word_indexjson.json();

  // console.log(model.summary());
  console.log("Model & Metadata Loaded Succesfully");
}
