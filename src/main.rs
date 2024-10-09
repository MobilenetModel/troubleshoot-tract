use tract_onnx::prelude::*;

fn main() {
    let model = tract_onnx::onnx()
        .model_for_path("saved_model.onnx")
        .unwrap()
        .into_optimized()
        .unwrap()
        .into_runnable()
        .unwrap();

    // open image, resize it and make a Tensor out of it
    let image = image::open("test_image.jpg").unwrap().to_rgb8();
    let resized = image::imageops::resize(&image, 224, 224, image::imageops::FilterType::Triangle);
    let image: Tensor = tract_ndarray::Array4::from_shape_fn((1, 224, 224, 3), |(_, y, x, c)| {
        f32::from(resized[(x.try_into().unwrap(), y.try_into().unwrap())][c]) / 255.0
    })
    .into();

    // run the model on the input
    let result = model.run(tvec!(image.into())).unwrap();

    // find and display the max value with its index
    let best = result[0]
        .to_array_view::<f32>()
        .unwrap()
        .iter()
        .copied()
        .zip(1..)
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    println!("result: {best:?}");
}
