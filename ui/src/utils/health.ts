export function healthDotClass(score: number | null | undefined): string {
  if (score == null) return 'health-dot--grey';
  if (score >= 0.75) return 'health-dot--green';
  if (score >= 0.60) return 'health-dot--yellow';
  return 'health-dot--red';
}
