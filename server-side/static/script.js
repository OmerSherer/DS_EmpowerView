const cameraButton = document.getElementById('camera');
const closeCameraButton = document.getElementById('close-camera');
const videoStream = document.getElementById('video-stream');
const fileInput = document.getElementById('file-upload');
const uploadButton = document.getElementById('upload-button');
const imageUpload = document.getElementById('imageUpload');

window.addEventListener("load",()=>{
    if(fileInput.value == "") {
        fileInput.innerHTML = "Choose file"
    }
})

async function openCamera(){
    const stream = await navigator.mediaDevices.getUserMedia({video: true, audio: false});
    videoStream.srcObject = stream;
    videoStream.play();
    cameraButton.style.display = 'none';
    closeCameraButton.style.display = 'block';
    uploadButton.style.display = 'none';
}

cameraButton.addEventListener("click" , openCamera)

imageTakeVideo.addEventListener("click" , openCamera)

closeCameraButton.onclick = () => {
    videoStream.srcObject.getTracks()[0].stop();
    videoStream.srcObject = null;
    closeCameraButton.style.display = 'none';
    cameraButton.style.display = 'block';
    if (fileInput.value) {
        uploadButton.style.display = 'block';
    }
}

fileInput.onchange = () => {
    if(fileInput.value.includes(".mp4")){
        uploadButton.style.visibility = 'visible';
    } else {
        alert("You need to insert mp4 file")
        uploadButton.style.visibility = 'hidden';
    }
   
}

// uploadButton.onclick = () => {
//     alert('Upload video successfully');
//     fileInput.value = '';
//     uploadButton.style.display = 'none';
// }

// imageUpload.addEventListener("click" , onclick)
