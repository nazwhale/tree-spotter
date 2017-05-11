require 'rails_helper'

feature 'homepage' do
  context 'displays a welcome message' do
    scenario "should say 'Welcome to Tree Spotter!'" do
      visit '/photos'
      expect(page).to have_content 'Welcome to Tree Spotter!'
      expect(page).to have_link 'Upload photo'
    end
  end

end
