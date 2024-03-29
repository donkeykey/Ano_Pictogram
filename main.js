let useFront = true;
let stream = null;
let stop_detect = false;
let net = null;
let whiteColorMode = true;
let drawCircleMask = false;
let videoWidth = 0;
let videoHeight = 0;

async function setupCamera(mode) {
	if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
		throw new Error(
			'Browser API navigator.mediaDevices.getUserMedia not available');
	}

	videoWidth = $('.main-contents').outerWidth() * 0.95;
	videoHeight = $('.main-contents').outerHeight() * 0.95;

	const video = document.getElementById('video');
	video.width = videoWidth;
	video.height = videoHeight;

	stream = await navigator.mediaDevices.getUserMedia({
		'audio': false,
		'video': {
			facingMode: mode,
			width: videoWidth,
			height: videoHeight,
		},
	});
	video.srcObject = stream;

	return new Promise((resolve) => {
		video.onloadedmetadata = () => {
			resolve(video);
		};
	});
}

function isAndroid() {
	return /Android/i.test(navigator.userAgent);
}

function isiOS() {
	return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

export function isMobile() {
	return isAndroid() || isiOS();
}

async function loadVideo(mode) {
	const video = await setupCamera(mode);
	video.play();
	stop_detect = false;

	return video;
}

const defaultQuantBytes = 2;

const defaultMobileNetMultiplier = isMobile() ? 0.50 : 0.75;
const defaultMobileNetStride = 16;
const defaultMobileNetInputResolution = 500;
const guiState = {
	input: {
		architecture: 'MobileNetV1',
		outputStride: defaultMobileNetStride,
		inputResolution: defaultMobileNetInputResolution,
		multiplier: defaultMobileNetMultiplier,
		quantBytes: defaultQuantBytes
	},
	singlePoseDetection: {
		minPoseConfidence: 0.1,
		minPartConfidence: 0.5,
	},
	multiPoseDetection: {
		maxPoseDetections: 5,
		minPoseConfidence: 0.15,
		minPartConfidence: 0.1,
		nmsRadius: 30.0,
	},
	output: {
		showVideo: false,
		showSkeleton: true,
		showPoints: true,
		showBoundingBox: false,
	},
	net: null,
};

function detectPoseInRealTime(video, net) {
	const canvas = document.getElementById('output');
	const ctx = canvas.getContext('2d');

	const flipPoseHorizontal = useFront;

	canvas.width = videoWidth;
	canvas.height = videoHeight;

	async function poseDetectionFrame() {
		if (guiState.changeToArchitecture) {
			// Important to purge variables and free up GPU memory
			guiState.net.dispose();
			toggleLoadingUI(true);
			guiState.net = await posenet.load({
				architecture: guiState.changeToArchitecture,
				outputStride: guiState.outputStride,
				inputResolution: guiState.inputResolution,
				multiplier: guiState.multiplier,
			});
			toggleLoadingUI(false);
			guiState.architecture = guiState.changeToArchitecture;
			guiState.changeToArchitecture = null;
		}

		if (guiState.changeToMultiplier) {
			guiState.net.dispose();
			toggleLoadingUI(true);
			guiState.net = await posenet.load({
				architecture: guiState.architecture,
				outputStride: guiState.outputStride,
				inputResolution: guiState.inputResolution,
				multiplier: +guiState.changeToMultiplier,
				quantBytes: guiState.quantBytes
			});
			toggleLoadingUI(false);
			guiState.multiplier = +guiState.changeToMultiplier;
			guiState.changeToMultiplier = null;
		}

		if (guiState.changeToOutputStride) {
			// Important to purge variables and free up GPU memory
			guiState.net.dispose();
			toggleLoadingUI(true);
			guiState.net = await posenet.load({
				architecture: guiState.architecture,
				outputStride: +guiState.changeToOutputStride,
				inputResolution: guiState.inputResolution,
				multiplier: guiState.multiplier,
				quantBytes: guiState.quantBytes
			});
			toggleLoadingUI(false);
			guiState.outputStride = +guiState.changeToOutputStride;
			guiState.changeToOutputStride = null;
		}

		if (guiState.changeToInputResolution) {
			// Important to purge variables and free up GPU memory
			guiState.net.dispose();
			toggleLoadingUI(true);
			guiState.net = await posenet.load({
				architecture: guiState.architecture,
				outputStride: guiState.outputStride,
				inputResolution: +guiState.changeToInputResolution,
				multiplier: guiState.multiplier,
				quantBytes: guiState.quantBytes
			});
			toggleLoadingUI(false);
			guiState.inputResolution = +guiState.changeToInputResolution;
			guiState.changeToInputResolution = null;
		}

		if (guiState.changeToQuantBytes) {
			// Important to purge variables and free up GPU memory
			guiState.net.dispose();
			toggleLoadingUI(true);
			guiState.net = await posenet.load({
				architecture: guiState.architecture,
				outputStride: guiState.outputStride,
				inputResolution: guiState.inputResolution,
				multiplier: guiState.multiplier,
				quantBytes: guiState.changeToQuantBytes
			});
			toggleLoadingUI(false);
			guiState.quantBytes = guiState.changeToQuantBytes;
			guiState.changeToQuantBytes = null;
		}

		let poses = [];
		let minPoseConfidence;
		let minPartConfidence;
		let all_poses = await guiState.net.estimatePoses(video, {
			flipHorizontal: flipPoseHorizontal,
			decodingMethod: 'multi-person',
			maxDetections: guiState.multiPoseDetection.maxPoseDetections,
			scoreThreshold: guiState.multiPoseDetection.minPartConfidence,
			nmsRadius: guiState.multiPoseDetection.nmsRadius
		});

		poses = poses.concat(all_poses);
		minPoseConfidence = +guiState.multiPoseDetection.minPoseConfidence;
		minPartConfidence = +guiState.multiPoseDetection.minPartConfidence;

		ctx.clearRect(0, 0, videoWidth, videoHeight);

		if (guiState.output.showVideo) {
			ctx.save();
			if (useFront) {
				ctx.scale(-1, 1);
				ctx.translate(-videoWidth, 0);
			}
			ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
			ctx.restore();
		} else {
			// draw background
			if (whiteColorMode) {
				ctx.fillStyle = "#FFFFFF";
			} else {
				ctx.fillStyle = "#001248";
			}
			ctx.fillRect(0, 0, canvas.width, canvas.height);
		}


		poses.forEach(({ score, keypoints }) => {
			if (score >= minPoseConfidence) {
				drawPictgram(keypoints, ctx);
			}
		});

		if (drawCircleMask) {
			$("canvas").css("border", "0px solid");
			ctx.globalCompositeOperation = 'destination-in';
			if (videoHeight < videoWidth) {
				ctx.arc(videoWidth/2, videoHeight/2, videoHeight/2, Math.PI * 2, 0, false);
			} else {
				ctx.arc(videoWidth/2, videoHeight/2, videoWidth/2, Math.PI * 2, 0, false);
			}
			ctx.fill();

			ctx.globalCompositeOperation = "source-over";
			ctx.lineWidth = 2;
			if (whiteColorMode) {
				ctx.strokeStyle = "#001248";
			} else {
				ctx.strokeStyle = "#FFFFFF";
			}
			ctx.beginPath();
			if (videoHeight < videoWidth) {
				ctx.arc(videoWidth/2, videoHeight/2, videoHeight/2 - 1, Math.PI * 2, 0, false);
			} else {
				ctx.arc(videoWidth/2, videoHeight/2, videoWidth/2 - 1, Math.PI * 2, 0, false);
			}
			ctx.stroke();
		} else {
			$("canvas").css("border", "2px solid");
			if (whiteColorMode) {
				$("canvas").css("border-color", "#001248")
			} else {
				$("canvas").css("border-color", "#FFFFFF")
			}
		}

		if (!stop_detect) {
			requestAnimationFrame(poseDetectionFrame);
		} else {
			console.log("stop detect");
		}
	}

	if (!stop_detect) {
		poseDetectionFrame();
	} else {
		console.log("stop detect");
	}
}

function drawPictgram(keypoints, ctx) {
	let color = "#001248";
	if (!whiteColorMode) {
		color = "#FFFFFF"
	}
	const headWeight = 1.2;
	const lineWeight = 1.2;

	// calc scale
	const scale = Math.pow((Math.pow((keypoints[1].position.x - keypoints[2].position.x), 2) + Math.pow((keypoints[1].position.y - keypoints[2].position.y), 2)), 0.5);

	// calc headSize
	const headSize = scale * headWeight;

	// calc lineWidth
	const lineWidth = scale * lineWeight;


	// nose
	ctx.beginPath();
	ctx.arc(keypoints[0].position.x, keypoints[0].position.y, headSize, 0, Math.PI * 2, true);
	ctx.fillStyle = color;
	ctx.fill();

	// 左肩
	ctx.beginPath();
	ctx.arc(keypoints[5].position.x, keypoints[5].position.y, lineWidth / 2, 0, Math.PI * 2, true);
	ctx.fillStyle = color;
	ctx.fill();

	// 右肩
	ctx.beginPath();
	ctx.arc(keypoints[6].position.x, keypoints[6].position.y, lineWidth / 2, 0, Math.PI * 2, true);
	ctx.fillStyle = color;
	ctx.fill();

	// 左手
	ctx.beginPath();
	ctx.arc(keypoints[9].position.x, keypoints[9].position.y, lineWidth / 2, 0, Math.PI * 2, true);
	ctx.fillStyle = color;
	ctx.fill();

	// 右手
	ctx.beginPath();
	ctx.arc(keypoints[10].position.x, keypoints[10].position.y, lineWidth / 2, 0, Math.PI * 2, true);
	ctx.fillStyle = color;
	ctx.fill();

	// 左足付け根
	ctx.beginPath();
	ctx.arc(keypoints[11].position.x, keypoints[11].position.y, lineWidth / 2, 0, Math.PI * 2, true);
	ctx.fillStyle = color;
	ctx.fill();

	// 右足付け根
	ctx.beginPath();
	ctx.arc(keypoints[12].position.x, keypoints[12].position.y, lineWidth / 2, 0, Math.PI * 2, true);
	ctx.fillStyle = color;
	ctx.fill();

	// 左足首
	ctx.beginPath();
	ctx.arc(keypoints[15].position.x, keypoints[15].position.y, lineWidth / 2, 0, Math.PI * 2, true);
	ctx.fillStyle = color;
	ctx.fill();

	// 右足首
	ctx.beginPath();
	ctx.arc(keypoints[16].position.x, keypoints[16].position.y, lineWidth / 2, 0, Math.PI * 2, true);
	ctx.fillStyle = color;
	ctx.fill();

	// 左腕
	ctx.beginPath();
	ctx.lineWidth = lineWidth;
	ctx.strokeStyle = color;
	ctx.moveTo(keypoints[5].position.x, keypoints[5].position.y);
	ctx.lineTo(keypoints[7].position.x, keypoints[7].position.y);
	ctx.lineTo(keypoints[9].position.x, keypoints[9].position.y);
	ctx.stroke();

	// 右腕
	ctx.beginPath();
	ctx.lineWidth = lineWidth;
	ctx.strokeStyle = color;
	ctx.moveTo(keypoints[6].position.x, keypoints[6].position.y);
	ctx.lineTo(keypoints[8].position.x, keypoints[8].position.y);
	ctx.lineTo(keypoints[10].position.x, keypoints[10].position.y);
	ctx.stroke();

	// 左足
	ctx.beginPath();
	ctx.lineWidth = lineWidth;
	ctx.strokeStyle = color;
	ctx.moveTo(keypoints[11].position.x, keypoints[11].position.y);
	ctx.lineTo(keypoints[13].position.x, keypoints[13].position.y);
	ctx.lineTo(keypoints[15].position.x, keypoints[15].position.y);
	ctx.stroke();

	// 右足
	ctx.beginPath();
	ctx.lineWidth = lineWidth;
	ctx.strokeStyle = color;
	ctx.moveTo(keypoints[12].position.x, keypoints[12].position.y);
	ctx.lineTo(keypoints[14].position.x, keypoints[14].position.y);
	ctx.lineTo(keypoints[16].position.x, keypoints[16].position.y);
	ctx.stroke();

	ctx.beginPath();
	ctx.stroke();
}

function syncCamera(video){
	useFront = !useFront;
	const mode = (useFront) ? "user" : { exact: "environment" };
	console.log(mode);
	stop_detect = true;
	if (stream !== null) {
		stream.getVideoTracks().forEach((camera) => {
			camera.stop();
		});
	}
	(async () => {
		let v = await loadVideo(mode);
		detectPoseInRealTime(video, guiState.net);
	})();
}

function reverseColor(){
	whiteColorMode = !whiteColorMode;
	if (whiteColorMode) {
		$("canvas").css("background-color", "#FFFFFF");
		$("body").css("background-color", "#FFFFFF")
		$("h6").css("color", "#001248")
		$("canvas").css("border-color", "#001248")
		$(".save-btn").css({
			"border-color": "#001248",
			"background-color": "#001248",
			"color": "#FFFFFF",
		})
	} else {
		$("canvas").css("background-color", "#001248");
		$("body").css("background-color", "#001248")
		$("h6").css("color", "#FFFFFF")
		$("canvas").css("border-color", "#FFFFFF")
		$(".save-btn").css({
			"border-color": "#dfdfdf",
			"background-color": "#dfdfdf",
			"color": "#001248",
		})
	}
}


(async () => {
	const net = await posenet.load({
		architecture: guiState.input.architecture,
		outputStride: guiState.input.outputStride,
		inputResolution: guiState.input.inputResolution,
		multiplier: guiState.input.multiplier,
		quantBytes: guiState.input.quantBytes
	});
	let video = await loadVideo('user');
	guiState.net = net;
	$('.save-btn').on('click', function() {
		let canvas = document.getElementById('output')
		var link = document.getElementById('hiddenLink');
	  	link.setAttribute('download', 'ano-pictogram.png');
	  	link.setAttribute('href', canvas.toDataURL("image/png").replace("image/png", "image/octet-stream"));
	  	link.click();
	});
	$('.cam-toggle-btn').on('click', function() {
		syncCamera(video);
	});
	$('.color-toggle-btn').on('click', function() {
		reverseColor();
	});
	$('.background-btn').on('click', function() {
		guiState.output.showVideo = !guiState.output.showVideo;
	});
	$('.frame-btn').on('click', function() {
		drawCircleMask = !drawCircleMask;
	});

	detectPoseInRealTime(video, net);
})();

