import axios from 'axios'

export async function getRecommendation(query) {
  const response = await axios.post('/recommend', { query })
  return response.data
}
