class Photo < ApplicationRecord
  has_attached_file :image, :styles => { :medium => "900x900>", :thumb => "300x300>" }, :default_url => "/images/:style/missing.png"
  validates_attachment_content_type :image, :content_type => /\Aimage\/.*\Z/

  require 'csv'

  def send_to_csv
    photo_path = 'images/its_a_three.png'
    CSV.open("selected_photo.csv", "wb") do |csv|
      csv << [photo_path]
    end
  end

  def run_NN
    result = exec("python ./MNIST_softmax_regression.py")
  end

  def get_prediction
    prediction = ""
    csv_text = File.read('NN_output.csv')
    CSV.foreach("NN_output.csv", "rb") do |row|
      prediction << row[0]
    end
    puts prediction
  end

end
