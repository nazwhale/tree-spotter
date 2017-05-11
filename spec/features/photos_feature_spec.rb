require 'rails_helper'

  feature 'photos' do
    context 'uploading photos' do
      scenario 'prompts user to choose a file to upload, then displays the photo' do
        visit '/photos'
        click_link 'Upload photo'
        click_button 'Submit'
        expect(current_path).to eq '/photos'
      end
    end
  end
