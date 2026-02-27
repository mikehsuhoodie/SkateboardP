using UnityEngine;

[RequireComponent(typeof(Rigidbody2D))]
public class SkateboardController : MonoBehaviour
{
    [Header("Movement")]
    public float moveSpeed = 5f;
    public float jumpForce = 10f;
    
    [Header("Magnetic Grind")]
    public float stickDistance = 0.5f;
    public float snapSpeed = 10f;
    public LayerMask trackLayer;
    
    private Rigidbody2D rb;
    private bool isGrinding = false;
    private bool isGrounded = false;
    private EdgeCollider2D currentTrack;
    
    void Awake()
    {
        rb = GetComponent<Rigidbody2D>();
    }

    void Update()
    {
        float moveInput = Input.GetAxisRaw("Horizontal");
        
        if (!isGrinding)
        {
            rb.velocity = new Vector2(moveInput * moveSpeed, rb.velocity.y);
            
            if (Input.GetButtonDown("Jump") && isGrounded)
            {
                rb.AddForce(Vector2.up * jumpForce, ForceMode2D.Impulse);
            }
        }

        // Grind Input (mapped to 'G' or secondary touch in future)
        if (Input.GetKeyDown(KeyCode.G) || Input.GetMouseButton(1))
        {
            TryStartGrind();
        }
        else if (Input.GetKeyUp(KeyCode.G) || Input.GetMouseButtonUp(1))
        {
            StopGrind();
        }

        if (isGrinding)
        {
            ApplyGrindPhysics();
        }
    }

    void TryStartGrind()
    {
        Collider2D hit = Physics2D.OverlapCircle(transform.position, stickDistance, trackLayer);
        if (hit != null && hit is EdgeCollider2D)
        {
            isGrinding = true;
            currentTrack = (EdgeCollider2D)hit;
            rb.gravityScale = 0; // Disable gravity while grinding
        }
    }

    void StopGrind()
    {
        if (isGrinding)
        {
            isGrinding = false;
            rb.gravityScale = 1;
            currentTrack = null;
        }
    }

    void ApplyGrindPhysics()
    {
        if (currentTrack == null) return;

        // Find the closest point on the edge collider
        Vector2 closestPoint = currentTrack.ClosestPoint(transform.position);
        
        // Snap to path
        transform.position = Vector3.Lerp(transform.position, (Vector3)closestPoint, Time.deltaTime * snapSpeed);

        // Basic movement along path (projected velocity)
        float moveInput = Input.GetAxisRaw("Horizontal");
        rb.velocity = transform.right * moveInput * moveSpeed; // Simplified: assumes sprite is oriented right
    }

    void OnCollisionStay2D(Collision2D collision)
    {
        if (collision.gameObject.layer == LayerMask.NameToLayer("Track"))
        {
            isGrounded = true;
        }
    }

    void OnCollisionExit2D(Collision2D collision)
    {
        if (collision.gameObject.layer == LayerMask.NameToLayer("Track"))
        {
            isGrounded = false;
        }
    }
}
