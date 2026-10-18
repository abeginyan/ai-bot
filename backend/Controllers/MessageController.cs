using Microsoft.AspNetCore.Mvc;

namespace Backend.Controllers;

[ApiController]
[Route("api/[controller]")]
public class MessageController : ControllerBase
{
    [HttpGet]
    public IActionResult Get()
    {
        return Ok(new MessageResponse(
            Message: "Hello from .NET 8 backend",
            Timestamp: DateTime.UtcNow));
    }
}

public record MessageResponse(string Message, DateTime Timestamp);
